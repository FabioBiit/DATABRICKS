from pyspark.sql import functions as f
from pyspark.sql import Window

# =========================
# Config
# =========================
N_USERS = 20000
N_SESSIONS_TARGET = 100000
P_ANON = 0.25
DAYS_BACK = 365

# =========================

objects = ["Zaino","Valigia","Portafoglio","Cintura","Borsone","Portadocumenti","Scarpe","Portachiavi"]
objects_arr = f.array(*[f.lit(o) for o in objects])

# (nome, pagine, peso, min_act, max_act, tipo_utente, desc)
JOURNEY_PATTERNS = [
    ("full_purchase",      ["Home","Search","Category","Product","Cart","Checkout"],                   0.12, 5, 6, "any",       "Acquisto completo con ricerca"),
    ("direct_purchase",    ["Home","Product","Cart","Checkout"],                                       0.06, 4, 4, "returning", "Acquisto diretto"),
    ("comparison_purchase",["Home","Category","Product","Product","Product","Cart","Checkout"],        0.08, 5, 7, "any",       "Confronto prodotti e acquisto"),
    ("browse_deep",        ["Home","Category","Product","Product","Category","Product"],               0.18, 4, 6, "any",       "Esplorazione"),
    ("browse_light",       ["Home","Category","Product"],                                              0.15, 2, 3, "new",       "Prima visita"),
    ("search_browse",      ["Home","Search","Product","Product"],                                      0.12, 3, 4, "any",       "Ricerca e valutazione"),
    ("cart_abandon_distracted",["Home","Category","Product","Cart","Category","Home"],                 0.05, 4, 6, "any",       "Abbandono - distrazione"),
    ("cart_abandon_price", ["Product","Cart","Home"],                                                  0.04, 3, 3, "any",       "Abbandono - prezzo"),
    ("support_general",    ["Home","Help","Blog"],                                                     0.05, 2, 3, "new",       "Info generali"),
    ("support_product",    ["Product","Help","Blog","Product"],                                        0.04, 3, 4, "any",       "Info prodotto"),
    ("support_checkout",   ["Cart","Help","Checkout"],                                                 0.02, 3, 3, "any",       "Aiuto checkout"),
    ("quick_search",       ["Home","Search","Product"],                                                0.06, 2, 3, "any",       "Ricerca veloce"),
    ("return_customer_direct",["Home","Product"],                                                      0.02, 2, 2, "power",     "Cliente abituale"),
    ("bounce_home",        ["Home"],                                                                   0.03, 1, 1, "any",       "Bounce home"),
    ("bounce_category",    ["Category"],                                                               0.02, 1, 1, "any",       "Bounce category"),
    ("bounce_search",      ["Search"],                                                                 0.01, 1, 1, "any",       "Bounce search"),
]

# =========================
# STEP 1: utenti + IP + visite (stocastico, no window)
# =========================
users = (
    spark.range(N_USERS)
         .withColumnRenamed("id","user_idx")
         .withColumn("Account", f.concat(f.lit("user"), f.lpad(f.col("user_idx").cast("string"), 6, "0")))
         .withColumn("h", f.crc32(f.col("Account")).cast("bigint"))
         .select(
             "user_idx","Account",
             f.concat_ws(".",
                 f.lit(10),
                 ((f.col("h") % 254) + 1).cast("int"),
                 (f.floor(f.col("h") / 254) % 254 + 1).cast("int"),
                 (f.floor(f.col("h") / 64516) % 254 + 1).cast("int")
             ).alias("Indirizzo_IP"),
             f.exp(f.randn(123)).alias("w")
         )
)

sum_w = users.agg(f.sum("w").alias("S")).first()["S"]
users = (users
    .withColumn("q_i", f.col("w")/f.lit(sum_w) * f.lit(N_SESSIONS_TARGET))
    .select(
        "user_idx","Account","Indirizzo_IP",
        (f.floor("q_i") + f.when(f.rand(777) < (f.col("q_i") - f.floor("q_i")), 1).otherwise(0)).cast("long").alias("visite_int")
    )
)
tot = users.agg(f.sum("visite_int").alias("tot")).first()["tot"]
delta = int(tot - N_SESSIONS_TARGET)
if delta != 0:
    k = abs(delta)
    adj = (users.select("user_idx").orderBy(f.rand(999)).limit(k).withColumn("bump", f.lit(-1 if delta > 0 else 1)))
    users = (users.join(adj, "user_idx", "left")
                  .select("user_idx","Account","Indirizzo_IP",
                          f.greatest(f.lit(0), f.col("visite_int") + f.coalesce(f.col("bump"), f.lit(0))).alias("visite_int"))
            )

users = users.withColumn("user_type",
    f.when(f.col("visite_int")==1,"new").when(f.col("visite_int").between(2,5),"returning").when(f.col("visite_int")>5,"power").otherwise("any")
)

# =========================
# STEP 2: sessioni
# =========================
sessions = (
    users.filter("visite_int > 0")
         .withColumn("session_num", f.explode(f.sequence(f.lit(1), f.col("visite_int"))))
         .select("user_idx","Account","Indirizzo_IP","user_type","session_num")
)

# =========================
# STEP 3: assegnazione journey (per user_type)
# =========================
patterns_df = spark.createDataFrame(
    [(n,p,w,mi,ma,ut,d) for (n,p,w,mi,ma,ut,d) in JOURNEY_PATTERNS],
    ["name","pages","weight","min_act","max_act","user_type","desc"]
)
# duplica i pattern 'any' su tutti i tipi
user_types = spark.createDataFrame([("any",),("new",),("returning",),("power",)], ["session_user_type"])
patterns_expanded = (patterns_df
    .join(user_types, (patterns_df.user_type == "any") | (patterns_df.user_type == user_types.session_user_type))
    .select(patterns_df["*"], user_types.session_user_type)
)

# normalizza pesi per session_user_type (DF piccolo -> window ok)
w = Window.partitionBy("session_user_type")
patterns_expanded = (patterns_expanded
    .withColumn("total_w", f.sum("weight").over(w))
    .withColumn("norm_w", f.col("weight")/f.col("total_w"))
)

# estrai un pattern per sessione con roulette wheel
sessions = sessions.withColumn("u", f.rand(555))
join_df = (sessions
    .join(patterns_expanded, sessions.user_type == patterns_expanded.session_user_type, "inner")
    .withColumn("cum_w", f.sum("norm_w").over(Window.partitionBy("user_idx","session_num").orderBy("name")))
    .where(f.col("u") <= f.col("cum_w"))
    .dropDuplicates(["user_idx","session_num"])
    .select(
        sessions["*"],
        f.col("name").alias("journey_name"),
        f.col("pages").alias("page_template"),
        "min_act","max_act"
    )
)

# num_actions:
# - per full_purchase: obbliga a coprire tutto il template (così Cart e Checkout sono garantiti)
# - altrimenti min..max uniforme intero
tpl_size = f.size("page_template")
join_df = (join_df
    .withColumn("num_actions",
        f.when(
            f.col("journey_name")=="full_purchase",
            tpl_size  # copri tutte le pagine del template
        ).otherwise(
            f.when(f.col("min_act")==f.col("max_act"), f.col("min_act"))
             .otherwise(f.col("min_act") + f.floor(f.rand(888) * (f.col("max_act") - f.col("min_act") + 1)))
        ).cast("int")
    )
    .drop("min_act","max_act")
)

# =========================
# STEP 4: esplodi azioni e mappa pagine
# =========================
actions = (join_df
    .withColumn("action_num", f.explode(f.sequence(f.lit(1), f.col("num_actions"))))
    .withColumn("template_size", f.size("page_template"))
    .withColumn("Pagina_Web",
        f.when(
            f.col("action_num") <= f.col("template_size"),
            f.element_at("page_template", f.col("action_num"))
        ).otherwise(
            # extra azioni: vocazione a ripetere le ultime pagine (Cart/Checkout se presenti)
            f.when(f.col("journey_name")=="full_purchase",
                   f.element_at("page_template", f.col("template_size"))  # tipicamente Checkout
            ).otherwise(
                f.element_at("page_template", f.col("template_size"))    # fallback: ultima pagina
            )
        )
    )
    .select("user_idx","Account","Indirizzo_IP","session_num","journey_name","Pagina_Web","action_num")
)

# =========================
# STEP 5: anonimizzazione (IP coerente per sessione)
# =========================
anon_hash = f.abs(f.crc32(f.concat_ws("|", f.lit("anon"), f.col("user_idx").cast("string"), f.col("session_num").cast("string"))))
anon_ip = f.concat_ws(".",
    f.lit(10),
    ((anon_hash % 254) + 1).cast("int"),
    ((f.floor(anon_hash / 254) % 254) + 1).cast("int"),
    ((f.floor(anon_hash / 64516) % 254) + 1).cast("int")
)
actions = (actions
    .withColumn("is_anon", f.rand(43) < f.lit(P_ANON))
    .withColumn("Indirizzo_IP", f.when(f.col("is_anon"), anon_ip).otherwise(f.col("Indirizzo_IP")))
    .withColumn("Account",      f.when(f.col("is_anon"), f.lit(None)).otherwise(f.col("Account")))
    .drop("is_anon")
)

# =========================
# STEP 6: timestamp + durate
# =========================
seconds_back = f.lit(DAYS_BACK*86400)
# start sessione random (uniforme su 365 gg)
actions = (actions
    .withColumn("session_start_sec", (f.rand(48)*seconds_back).cast("long"))
    .withColumn("session_start_unix", f.unix_timestamp() - f.col("session_start_sec"))
)

# durate per pagina (min,max) -> calcolo leggero
duration_map = {
    "Home": (8, 90),
    "Search": (10, 120),
    "Category": (15, 150),
    "Product": (25, 300),
    "Cart": (20, 180),
    "Checkout": (45, 360),
    "Help": (30, 240),
    "Blog": (40, 300)
}
dur_expr = None
for pg, (mn, mx) in duration_map.items():
    cur = f.round(f.lit(mn) + f.rand(100 + hash(pg) % 50) * (mx - mn), 2)
    if dur_expr is None:
        dur_expr = f.when(f.col("Pagina_Web") == pg, cur)
    else:
        dur_expr = dur_expr.when(f.col("Pagina_Web") == pg, cur)
Durata_Visita_Sito = dur_expr.otherwise(f.lit(30.0))

actions = (actions
    .withColumn("Durata_Visita_Sito", Durata_Visita_Sito)
    .withColumn("Data_Invio_Statistiche",
        f.to_date(f.from_unixtime(f.col("session_start_unix") + (f.col("action_num")-1)*f.col("Durata_Visita_Sito"))))
    .drop("session_start_sec","session_start_unix")
)

# =========================
# STEP 7: Oggetto_Visionato
#  - Oggetto "sticky" per sessione per le pagine di consulto
#  - Cart/Checkout DEVONO usare un oggetto tra quelli cercati nella stessa sessione
# =========================
from pyspark.sql import Window

obj_hash = f.abs(f.crc32(f.concat_ws("|", f.col("user_idx").cast("string"), f.col("session_num").cast("string"))))
session_obj_idx = (obj_hash % len(objects)).cast("int")
session_obj = f.element_at(objects_arr, session_obj_idx + f.lit(1))

# 1) Assegna Oggetto_Visionato "di base" per le pagine che possono avere un oggetto
actions = (actions
    .withColumn(
        "Oggetto_Visionato",
        f.when(
            f.col("Pagina_Web").isin("Product","Category","Cart","Checkout","Search"),
            # per le pagine di consulto (Product/Category/Search): 80% sticky, 20% random
            f.when(
                f.col("Pagina_Web").isin("Product","Category","Search"),
                f.when(f.rand(202)<0.80, session_obj)
                 .otherwise(f.element_at(objects_arr, (f.rand(203)*len(objects)).cast("int")+1))
            )
            # placeholder temporaneo per Cart/Checkout (lo correggiamo sotto in base ai "cercati")
            .otherwise(session_obj)
        )
    )
)

# 2) Costruisci l'insieme dei prodotti "cercati" nella sessione (Product/Search/Category)
w_sess = Window.partitionBy("user_idx","session_num")
actions = (actions
    .withColumn(
        "searched_set",
        f.array_distinct(
            f.collect_set(
                f.when(
                    f.col("Pagina_Web").isin("Product","Search","Category"),
                    f.col("Oggetto_Visionato")
                )
            ).over(w_sess)
        )
    )
)

# 3) Forza Cart/Checkout a prendere un oggetto "cercato" nella stessa sessione
#    (se la lista è vuota, fallback al session_obj)
searched_size = f.size("searched_set")
rand_idx = (f.floor(f.rand(204)*f.greatest(searched_size, f.lit(1))) + 1).cast("int")  # indice 1-based
chosen_from_searched = f.element_at("searched_set", rand_idx)

actions = (actions
    .withColumn(
        "Oggetto_Visionato",
        f.when(
            f.col("Pagina_Web").isin("Cart","Checkout"),
            f.when(searched_size > 0, chosen_from_searched).otherwise(session_obj)
        ).otherwise(f.col("Oggetto_Visionato"))
    )
    .withColumn(
        "Tempo_Visita_Su_Oggetto",
        f.when(
            f.col("Oggetto_Visionato").isNotNull(),
            f.round(f.least(f.col("Durata_Visita_Sito")*(0.25+f.rand(47)*0.65), f.col("Durata_Visita_Sito")), 2)
        ).otherwise(f.lit(0.00))
    )
    .drop("searched_set")  # colonna ausiliaria non più necessaria
)


# =========================
# STEP 8: ID_Sessione (hash deterministico)
# =========================
session_key = f.concat_ws("|",
    f.coalesce(f.col("Account"), f.concat(f.lit("anon#"), f.col("user_idx"))),
    f.col("Indirizzo_IP"),
    f.col("session_num").cast("string")
)
actions = actions.withColumn("ID_Sessione", f.abs(f.crc32(session_key)).cast("bigint"))

# =========================
# STEP 9: Output
# =========================
out_df = (actions
    .select(
        "ID_Sessione",
        "action_num",
        "journey_name",
        f.col("Pagina_Web").cast("string").alias("Pagina_Web"),
        f.col("Account").cast("string").alias("Account"),
        f.col("Durata_Visita_Sito").cast("decimal(10,2)").alias("Durata_Visita_Sito"),
        f.col("Oggetto_Visionato").cast("string").alias("Oggetto_Visionato"),
        f.col("Tempo_Visita_Su_Oggetto").cast("decimal(10,2)").alias("Tempo_Visita_Su_Oggetto"),
        f.col("Data_Invio_Statistiche").cast("date").alias("Data_Invio_Statistiche"),
        f.col("Indirizzo_IP").cast("string").alias("Indirizzo_IP")
    )
    .repartition(8, "Data_Invio_Statistiche")
)

(out_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("Data_Invio_Statistiche")
    .saveAsTable("default.row_data_site"))

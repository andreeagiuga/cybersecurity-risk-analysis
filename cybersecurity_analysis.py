import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import statsmodels.api as sm

# =============================================================================
# CONFIGURARE GLOBALA
# =============================================================================

TITLU_FIRMA = 'Andreea & Ruxandra Cybersecurity SRL - Analiza de piata'

REGIUNI = {
    'Europa de Nord': ['Denmark', 'Estonia', 'Finland', 'Latvia', 'Lithuania', 'Sweden'],
    'Europa de Vest': ['Austria', 'Belgium', 'France', 'Germany', 'Ireland', 'Luxembourg', 'Netherlands'],
    'Europa de Sud':  ['Croatia', 'Cyprus', 'Greece', 'Italy', 'Malta', 'Portugal', 'Slovenia', 'Spain'],
    'Europa de Est':  ['Bulgaria', 'Czechia', 'Hungary', 'Poland', 'Romania', 'Slovakia'],
    'Tari Candidate': ['Bosnia and Herzegovina', 'North Macedonia', 'Serbia', 'Türkiye']
}

def get_regiune(tara):
    for regiune, lista_tari in REGIUNI.items():
        if tara in lista_tari:
            return regiune
    return 'Necunoscut'

INDICATORI_RISC = [
    'frauda_card', 'furt_identitate', 'phishing', 'pharming',
    'abuz_date_personale', 'cont_spart', 'pierdere_date_virus', 'pierdere_financiara'
]

INDICATORI_FRAUDA   = ['frauda_card', 'phishing', 'pharming', 'pierdere_financiara', 'scor_risc_total']
INDICATORI_NUMERICI = INDICATORI_RISC + ['competente_avansate', 'utilizare_banking_online', 'scor_risc_total']


# =============================================================================
# 1. IMPORT DATE
# =============================================================================

df = pd.read_csv('date_proiect_ps.csv')

print(f"Dimensiune initiala: {df.shape[0]} inregistrari x {df.shape[1]} coloane")
print(f"\nPrimele 5 inregistrari:\n{df.head()}")
print(f"\nTipuri de date:\n{df.dtypes}")


# =============================================================================
# 2. TRATAREA VALORILOR LIPSA
# =============================================================================

col_tara = df.columns[0]
print(f"\nValori lipsa inainte de tratare: {df.isnull().sum().sum()}")

df = df.dropna(how='all').reset_index(drop=True)
print(f"Valori lipsa dupa eliminarea randului gol: {df.isnull().sum().sum()}")

for col in df.select_dtypes(include='number').columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(round(df[col].mean(), 2))

print(f"Valori lipsa dupa completare cu media: {df.isnull().sum().sum()}")


# =============================================================================
# 3. STERGEREA INREGISTRARILOR SI COLOANELOR IRELEVANTE
# =============================================================================

tari_de_sters = [
    'European Union - 28 countries (2013-2020)',
    'United Kingdom', 'Iceland', 'Norway', 'Switzerland'
]

df = df[~df[col_tara].isin(tari_de_sters)].reset_index(drop=True)
df = df.drop(columns=df.columns[9])

print(f"Inregistrari dupa curatare: {df.shape[0]} tari | {df.shape[1]} coloane")


# =============================================================================
# 4. REDENUMIRE COLOANE SI CREARE INDICATORI NOI
# =============================================================================

cols = list(df.columns)
rename_map = {
    cols[0]:  'tara',           cols[1]:  'frauda_card',
    cols[2]:  'furt_identitate',cols[3]:  'phishing',
    cols[4]:  'pharming',       cols[5]:  'abuz_date_personale',
    cols[6]:  'cont_spart',     cols[7]:  'pierdere_date_virus',
    cols[8]:  'pierdere_financiara', cols[9]: 'competente_scazute',
    cols[10]: 'competente_de_baza', cols[11]: 'competente_avansate',
    cols[12]: 'fara_competente',    cols[13]: 'utilizare_email',
    cols[14]: 'utilizare_info_produse', cols[15]: 'utilizare_retele_sociale',
    cols[16]: 'utilizare_banking_online'
}
df = df.rename(columns=rename_map)

df['scor_risc_total'] = df[INDICATORI_RISC].sum(axis=1).round(2)
df['categorie_piata'] = df['scor_risc_total'].apply(
    lambda s: 'Prioritate Mare' if s >= 50 else ('Prioritate Medie' if s >= 25 else 'Prioritate Mica')
)
df['regiune'] = df['tara'].apply(get_regiune)

print(f"\nTop 10 tari dupa scorul de risc:")
print(df[['tara', 'scor_risc_total', 'categorie_piata']]
      .sort_values('scor_risc_total', ascending=False).head(10).to_string(index=False))

df.to_csv('cybershield_date_pregatite.csv', index=False)
print("\nFisier salvat: cybershield_date_pregatite.csv")


# =============================================================================
# 5. LISTE SI DICTIONARE
# =============================================================================

print("\nNumarul de tari per regiune:")
for regiune, tari in REGIUNI.items():
    print(f"  {regiune}: {len(tari)} tari")

media_risc = df['scor_risc_total'].mean()
tari_peste_medie = sorted(df[df['scor_risc_total'] > media_risc]['tara'].tolist())
print(f"\nMedia scorului de risc european: {media_risc:.2f}")
print(f"Tari cu scor peste medie ({len(tari_peste_medie)}): {tari_peste_medie}")

medie_pe_regiune = {
    reg: round(df[df['regiune'] == reg]['scor_risc_total'].mean(), 2)
    for reg in REGIUNI
}
print("\nMedia scorului de risc per regiune:")
for reg, medie in sorted(medie_pe_regiune.items(), key=lambda x: x[1], reverse=True):
    print(f"  {reg}: {medie}")


# =============================================================================
# 6. SETURI SI TUPLURI
# =============================================================================

praguri      = (20.0, 3.0, 10.0, 1.0)
nume_praguri = ('phishing > 20%', 'frauda_card > 3%', 'pharming > 10%', 'pierdere_financiara > 1%')

print("\nPraguri de alerta (tuplu fix):")
for nume, val in zip(nume_praguri, praguri):
    print(f"  {nume} (prag: {val})")

set_phishing     = set(df[df['phishing']           > praguri[0]]['tara'])
set_frauda_card  = set(df[df['frauda_card']         > praguri[1]]['tara'])
set_pharming     = set(df[df['pharming']            > praguri[2]]['tara'])
set_pierdere_fin = set(df[df['pierdere_financiara'] > praguri[3]]['tara'])

set_risc_multiplu = set_phishing & set_frauda_card & set_pharming & set_pierdere_fin
set_orice_risc    = set_phishing | set_frauda_card | set_pharming | set_pierdere_fin

print(f"\nTari expuse la toate 4 riscuri ({len(set_risc_multiplu)}): {sorted(set_risc_multiplu)}")
print(f"Tari cu cel putin un risc ridicat ({len(set_orice_risc)}): {sorted(set_orice_risc)}")
print(f"Romania in risc multiplu: {'Romania' in set_risc_multiplu}")


# =============================================================================
# 7. ACCESAREA DATELOR — loc si iloc
# =============================================================================

print("\nPrimele 3 inregistrari, primele 4 coloane (iloc):")
print(df.iloc[0:3, 0:4])

print("\nDate complete Romania (loc):")
print(df.loc[df['tara'] == 'Romania'].to_string(index=False))

print("\nTari din Europa de Est (loc):")
print(df.loc[df['regiune'] == 'Europa de Est',
             ['tara', 'phishing', 'frauda_card', 'scor_risc_total', 'categorie_piata']]
      .to_string(index=False))

print("\nTop 5 tari dupa scor de risc (iloc dupa sortare):")
df_sortat = df.sort_values('scor_risc_total', ascending=False).reset_index(drop=True)
print(df_sortat.iloc[0:5][['tara', 'regiune', 'scor_risc_total', 'categorie_piata']])


# =============================================================================
# 8. FUNCTII
# =============================================================================

def calculeaza_scor_digitalizare(row):
    indicatori = ['utilizare_email', 'utilizare_info_produse',
                  'utilizare_retele_sociale', 'utilizare_banking_online']
    return round(sum(row[ind] for ind in indicatori) / len(indicatori), 2)

def recomanda_serviciu(scor_risc, scor_digitalizare, prag_risc=40, prag_digital=40):
    if scor_risc >= prag_risc and scor_digitalizare >= prag_digital:
        return 'Pachet Enterprise complet'
    elif scor_risc >= prag_risc:
        return 'Solutii de protectie + Training digital'
    elif scor_digitalizare >= prag_digital:
        return 'Audit de securitate preventiv'
    else:
        return 'Training de awareness cibernetic'

def genereaza_raport_tara(tara, df):
    row = df[df['tara'] == tara].iloc[0]
    return f"""
{'─' * 40}
RAPORT PIATA: {tara.upper()}
{'─' * 40}
  Regiune:             {row['regiune']}
  Scor risc:           {row['scor_risc_total']}
  Categorie piata:     {row['categorie_piata']}
  Phishing:            {row['phishing']}%
  Frauda card:         {row['frauda_card']}%
  Banking online:      {row['utilizare_banking_online']}%
  Serviciu recomandat: {row['serviciu_recomandat']}
{'─' * 40}"""

df['scor_digitalizare']   = df.apply(calculeaza_scor_digitalizare, axis=1)
df['serviciu_recomandat'] = df.apply(
    lambda row: recomanda_serviciu(row['scor_risc_total'], row['scor_digitalizare']), axis=1
)

print("\nServiciu recomandat per tara:")
print(df[['tara', 'scor_risc_total', 'scor_digitalizare', 'serviciu_recomandat']]
      .sort_values('scor_risc_total', ascending=False).to_string(index=False))

print(genereaza_raport_tara('Romania', df))


# =============================================================================
# 9. STRUCTURI CONDITIONALE
# =============================================================================

def maturitate_digitala(row):
    banking, competente = row['utilizare_banking_online'], row['competente_avansate']
    if banking > 70 and competente > 35:
        return 'Piata matura - solutii enterprise avansate'
    elif banking > 50 and competente > 25:
        return 'Piata in dezvoltare - solutii standard'
    elif banking > 30 or competente > 20:
        return 'Piata emergenta - solutii entry-level + training'
    else:
        return 'Piata incipient digitalizata - training prioritar'

df['maturitate_digitala'] = df.apply(maturitate_digitala, axis=1)

ro = df[df['tara'] == 'Romania'].iloc[0]
if ro['categorie_piata'] == 'Prioritate Mare' and ro['scor_digitalizare'] > 30:
    decizie = 'INTRA IMEDIAT - piata prioritara cu digitalizare suficienta'
elif ro['categorie_piata'] == 'Prioritate Mare':
    decizie = 'INTRA CU PRECAUTIE - risc mare dar digitalizare scazuta'
elif ro['categorie_piata'] == 'Prioritate Medie':
    decizie = 'MONITORIZEAZA - piata de interes mediu'
else:
    decizie = 'ASTEAPTA - piata cu potential scazut pe termen scurt'

print(f"\nDecizie intrare piata Romania: {decizie}")
print(f"\nDistributia maturitatii digitale:\n{df['maturitate_digitala'].value_counts()}")


# =============================================================================
# 10. STRUCTURI REPETITIVE
# =============================================================================

print("\nRAPORT ALERTE CIBERNETICE:")
print('─' * 55)
nr_alerte = 0
for _, row in df.iterrows():
    alerte = []
    if row['phishing']            > 20: alerte.append(f"phishing {row['phishing']:.1f}%")
    if row['frauda_card']         > 3:  alerte.append(f"frauda card {row['frauda_card']:.1f}%")
    if row['pharming']            > 10: alerte.append(f"pharming {row['pharming']:.1f}%")
    if row['pierdere_financiara'] > 1:  alerte.append(f"pierdere fin. {row['pierdere_financiara']:.1f}%")
    if alerte:
        nr_alerte += 1
        print(f"  [{nr_alerte:02d}] {row['tara']:<25} ALERTE: {', '.join(alerte)}")

print(f"\nTotal tari cu cel putin o alerta: {nr_alerte} din {len(df)}")

print("\nTop piata per regiune:")
for idx, regiune in enumerate(REGIUNI.keys(), start=1):
    subset = df[df['regiune'] == regiune]
    if not subset.empty:
        top = subset.loc[subset['scor_risc_total'].idxmax()]
        print(f"  {idx}. {regiune:<20} -> {top['tara']} (scor: {top['scor_risc_total']:.2f})")

df.to_csv('cybershield_date_pregatite.csv', index=False)


# =============================================================================
# 11. GROUPBY
# =============================================================================

print("\nMedia indicatorilor de frauda per regiune:")
print(df.groupby('regiune')[INDICATORI_FRAUDA].mean().round(2)
      .sort_values('scor_risc_total', ascending=False).to_string())

print("\nAgregare (scor_risc_total) per regiune:")
agregare = df.groupby('regiune')['scor_risc_total'].agg(['mean', 'max', 'min', 'std', 'count']).round(2)
agregare.columns = ['medie', 'maxim', 'minim', 'std_dev', 'nr_tari']
print(agregare.sort_values('medie', ascending=False).to_string())

print("\nRomania vs media Europei de Est:")
media_est = df[df['regiune'] == 'Europa de Est'][INDICATORI_FRAUDA].mean().round(2)
romania   = df[df['tara'] == 'Romania'][INDICATORI_FRAUDA].iloc[0]
print(pd.DataFrame({'Romania': romania, 'Media Europa de Est': media_est}).to_string())


# =============================================================================
# 12. STATISTICI DESCRIPTIVE
# =============================================================================

print("\nStatistici descriptive (indicatori cheie):")
print(df[INDICATORI_NUMERICI].describe().round(2).to_string())

medie_eu  = df[INDICATORI_NUMERICI].mean()
ro_vals   = df[df['tara'] == 'Romania'][INDICATORI_NUMERICI].iloc[0]
diferente = (ro_vals - medie_eu).round(2)

print("\nRomania fata de media europeana:")
for col, val in diferente.items():
    semn = '+' if val > 0 else ''
    print(f"  {col:<30} {semn}{val:.2f}")

print("\nMatricea de corelatie:")
cols_corr = ['phishing', 'frauda_card', 'pharming', 'competente_avansate',
             'utilizare_banking_online', 'scor_risc_total']
print(df[cols_corr].corr().round(2).to_string())


# =============================================================================
# 13. MERGE / JOIN
# =============================================================================

df_prezenta = pd.DataFrame({
    'tara': ['Romania', 'Germany', 'France', 'Spain', 'Italy',
             'Netherlands', 'Belgium', 'Austria', 'Sweden', 'Denmark'],
    'are_prezenta':        ['Da'] * 10,
    'venituri_anuale_eur': [850000, 3200000, 4100000, 1900000, 2100000,
                             3500000, 2200000, 1700000, 2800000, 1500000],
    'nr_clienti_activi':   [45, 180, 210, 95, 115, 175, 110, 85, 140, 75]
})

df_inner = pd.merge(df, df_prezenta, on='tara', how='inner')
print("\nPiete cu prezenta activa (inner join):")
print(df_inner[['tara', 'scor_risc_total', 'categorie_piata', 'venituri_anuale_eur']]
      .sort_values('venituri_anuale_eur', ascending=False).to_string(index=False))

df_left = pd.merge(df, df_prezenta, on='tara', how='left')
df_left['are_prezenta']        = df_left['are_prezenta'].fillna('Nu')
df_left['venituri_anuale_eur'] = df_left['venituri_anuale_eur'].fillna(0)

gap = df_left[(df_left['categorie_piata'] == 'Prioritate Mare') & (df_left['are_prezenta'] == 'Nu')]
print(f"\nGap analysis — Prioritate Mare fara prezenta ({len(gap)} tari):")
print(gap[['tara', 'regiune', 'scor_risc_total']].sort_values('scor_risc_total', ascending=False)
      .to_string(index=False))

venit_mediu = df_inner['venituri_anuale_eur'].mean()
print(f"\nPotential venituri neexploatate: ~{len(gap) * venit_mediu:,.0f} EUR/an")

df_left.to_csv('cybershield_date_complete.csv', index=False)


# =============================================================================
# 14. VIZUALIZARE GRAFICA
# =============================================================================

MEDIA_EU = df['scor_risc_total'].mean()

# Graficul 1: Top 10 tari dupa scorul de risc
top10  = df.nlargest(10, 'scor_risc_total').sort_values('scor_risc_total')
culori = ['#d73027' if t == 'Romania' else '#4393c3' for t in top10['tara']]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top10['tara'], top10['scor_risc_total'], color=culori)
for bar, val in zip(bars, top10['scor_risc_total']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}', va='center', fontsize=9)
ax.axvline(x=MEDIA_EU, color='orange', linestyle='--', linewidth=1.5,
           label=f'Media europeana ({MEDIA_EU:.1f})')
ax.set_xlabel('Scor de risc cibernetic')
ax.set_title(f'Top 10 tari dupa scorul de risc cibernetic\n{TITLU_FIRMA}', fontsize=12)
ax.legend()
ax.set_xlim(0, 90)
plt.tight_layout()
plt.savefig('grafic1_top10_risc.png', dpi=150, bbox_inches='tight')
plt.show()

# Graficul 2: Scatter — Competente digitale vs Phishing
culori_regiune = {
    'Europa de Nord': '#2166ac', 'Europa de Vest': '#4dac26',
    'Europa de Sud':  '#d7191c', 'Europa de Est':  '#fdae61',
    'Tari Candidate': '#999999'
}

fig, ax = plt.subplots(figsize=(10, 7))
for regiune, culoare in culori_regiune.items():
    subset = df[df['regiune'] == regiune]
    ax.scatter(subset['competente_avansate'], subset['phishing'],
               c=culoare, label=regiune, s=80, alpha=0.8, edgecolors='white')
    for _, row in subset.iterrows():
        ax.annotate(row['tara'], (row['competente_avansate'], row['phishing']),
                    fontsize=7, textcoords='offset points', xytext=(4, 2))
ax.axhline(y=df['phishing'].mean(), color='gray', linestyle=':', alpha=0.7)
ax.axvline(x=df['competente_avansate'].mean(), color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Competente digitale avansate (%)')
ax.set_ylabel('Rata de phishing (%)')
ax.set_title(f'Relatia dintre competente digitale si phishing\n{TITLU_FIRMA}', fontsize=12)
ax.legend(title='Regiune', loc='upper left')
plt.tight_layout()
plt.savefig('grafic2_scatter_competente_phishing.png', dpi=150, bbox_inches='tight')
plt.show()

# Graficul 3: Pie chart — Distributia pietelor pe categorii
distributie = df['categorie_piata'].value_counts()

fig, ax = plt.subplots(figsize=(8, 7))
wedges, _, autotexts = ax.pie(
    distributie.values, labels=distributie.index, autopct='%1.1f%%',
    colors=['#d73027', '#fdae61', '#4daf4a'], explode=[0.05] * len(distributie),
    startangle=90, textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontweight('bold')
legend_labels = [f'{cat} ({cnt} tari)' for cat, cnt in zip(distributie.index, distributie.values)]
ax.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), fontsize=10)
ax.set_title(f'Distributia pietelor pe categorii de prioritate\n{TITLU_FIRMA}', fontsize=12)
plt.tight_layout()
plt.savefig('grafic3_pie_categorii.png', dpi=150, bbox_inches='tight')
plt.show()

# Graficul 4: Romania vs Media UE
indicatori = ['phishing', 'pharming', 'frauda_card', 'competente_avansate', 'utilizare_banking_online']
etichete   = ['Phishing\n(%)', 'Pharming\n(%)', 'Frauda\ncard (%)', 'Competente\navansate (%)', 'Banking\nonline (%)']
romania_vals  = df[df['tara'] == 'Romania'][indicatori].iloc[0].values
media_eu_vals = df[indicatori].mean().values
x, latime = range(len(indicatori)), 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar([i - latime / 2 for i in x], romania_vals,  latime, label='Romania',        color='#d73027', alpha=0.85)
bars2 = ax.bar([i + latime / 2 for i in x], media_eu_vals, latime, label='Media europeana', color='#4393c3', alpha=0.85)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color='#d73027')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color='#4393c3')
ax.set_xticks(list(x))
ax.set_xticklabels(etichete, fontsize=10)
ax.set_ylabel('Valoare (%)')
ax.set_title(f'Romania vs Media Europeana — Indicatori cheie\n{TITLU_FIRMA}', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, max(romania_vals.max(), media_eu_vals.max()) * 1.2)
plt.tight_layout()
plt.savefig('grafic4_romania_vs_medie.png', dpi=150, bbox_inches='tight')
plt.show()

# Graficul 5: Media scorului de risc per regiune
medie_regiune = df.groupby('regiune')['scor_risc_total'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(medie_regiune.index, medie_regiune.values,
              color=['#d73027', '#f46d43', '#fdae61', '#a6d96a', '#66bd63'],
              edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, medie_regiune.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=MEDIA_EU, color='navy', linestyle='--', linewidth=1.5,
           label=f'Media generala ({MEDIA_EU:.1f})')
ax.set_xlabel('Regiune')
ax.set_ylabel('Scor mediu de risc')
ax.set_title(f'Scorul mediu de risc per regiune europeana\n{TITLU_FIRMA}', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 65)
plt.tight_layout()
plt.savefig('grafic5_risc_per_regiune.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 15. K-MEANS CLUSTERING
# =============================================================================

cols_cluster = ['frauda_card', 'phishing', 'pharming',
                'pierdere_financiara', 'competente_avansate', 'utilizare_banking_online']

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[cols_cluster].values)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

medii_risc = df.groupby('cluster')['scor_risc_total'].mean().sort_values(ascending=False)
etichete_cluster = {
    medii_risc.index[0]: 'Cluster A - Risc Ridicat',
    medii_risc.index[1]: 'Cluster B - Risc Mediu',
    medii_risc.index[2]: 'Cluster C - Risc Scazut'
}
df['profil_piata'] = df['cluster'].map(etichete_cluster)

print("Segmentarea pietelor prin K-Means:")
for cluster_id, eticheta in etichete_cluster.items():
    tari     = df[df['cluster'] == cluster_id]['tara'].tolist()
    scor_med = df[df['cluster'] == cluster_id]['scor_risc_total'].mean()
    print(f"\n  {eticheta} (scor mediu: {scor_med:.2f}):\n    {', '.join(tari)}")

culori_cluster = {
    'Cluster A - Risc Ridicat': '#d73027',
    'Cluster B - Risc Mediu':   '#fdae61',
    'Cluster C - Risc Scazut':  '#4dac26'
}

fig, ax = plt.subplots(figsize=(10, 7))
for profil, culoare in culori_cluster.items():
    subset = df[df['profil_piata'] == profil]
    ax.scatter(subset['utilizare_banking_online'], subset['phishing'],
               c=culoare, label=profil, s=100, alpha=0.85, edgecolors='white')
    for _, row in subset.iterrows():
        ax.annotate(row['tara'], (row['utilizare_banking_online'], row['phishing']),
                    fontsize=7, textcoords='offset points', xytext=(4, 2))
ax.set_xlabel('Utilizare banking online (%)')
ax.set_ylabel('Rata phishing (%)')
ax.set_title('Segmentarea pietelor europene prin K-Means Clustering', fontsize=12)
ax.legend(title='Profil piata', loc='upper left')
plt.tight_layout()
plt.savefig('grafic_clustering.png', dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# 16. REGRESIE LOGISTICA
# =============================================================================

df['banking_ridicat'] = (df['utilizare_banking_online'] > 50).astype(int)

X = df[['phishing', 'frauda_card', 'pharming', 'competente_avansate', 'fara_competente']].values
y = df['banking_ridicat'].values

scaler_lr   = StandardScaler()
X_scaled_lr = scaler_lr.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_lr, y, test_size=0.3, random_state=42)

model_lr = LogisticRegression(random_state=42, max_iter=200)
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

print(f"\nRegresie Logistica — Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=['Banking scazut', 'Banking ridicat']))
print(f"Matricea de confuzie:\n{confusion_matrix(y_test, y_pred)}")

coef_df = pd.DataFrame({
    'Variabila':  ['phishing', 'frauda_card', 'pharming', 'competente_avansate', 'fara_competente'],
    'Coeficient': model_lr.coef_[0].round(3)
}).sort_values('Coeficient', ascending=False)
print(f"\nCoeficientii modelului:\n{coef_df.to_string(index=False)}")

ro_features = df[df['tara'] == 'Romania'][
    ['phishing', 'frauda_card', 'pharming', 'competente_avansate', 'fara_competente']].values
ro_prob = model_lr.predict_proba(scaler_lr.transform(ro_features))[0][1]
ro_pred = model_lr.predict(scaler_lr.transform(ro_features))[0]
print(f"\nPredictie Romania: {ro_prob:.2%} probabilitate banking ridicat "
      f"-> {'Banking ridicat' if ro_pred == 1 else 'Banking scazut'}")


# =============================================================================
# 17. REGRESIE MULTIPLA OLS
# =============================================================================

Y     = df['phishing']
X_vars = df[['frauda_card', 'pharming', 'competente_avansate',
              'utilizare_banking_online', 'utilizare_retele_sociale']]
X_sm  = sm.add_constant(X_vars)

model_ols = sm.OLS(Y, X_sm).fit()
print("\nRegresie Multipla OLS (variabila dependenta: phishing):")
print(model_ols.summary())

rezultate = pd.DataFrame({
    'Coeficient':            model_ols.params.round(4),
    'P-value':               model_ols.pvalues.round(4),
    'Semnificativ (p<0.05)': model_ols.pvalues < 0.05
})
print(f"\nCoeficienti si semnificatie:\n{rezultate.to_string()}")

ro_x             = X_sm[df['tara'] == 'Romania']
ro_pred_phishing = model_ols.predict(ro_x).values[0]
ro_actual        = df[df['tara'] == 'Romania']['phishing'].values[0]
print(f"\nValidare Romania: actual={ro_actual:.2f}% | prezis={ro_pred_phishing:.2f}% "
      f"| eroare={abs(ro_actual - ro_pred_phishing):.2f}%")

y_fitted = model_ols.fittedvalues
fig, ax  = plt.subplots(figsize=(9, 6))
ax.scatter(Y, y_fitted, color='#4393c3', alpha=0.7, s=80, edgecolors='white')
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linestyle='--',
        linewidth=1.5, label='Linia perfecta (y=x)')
ro_fitted = y_fitted[df['tara'] == 'Romania'].values[0]
ax.scatter(ro_actual, ro_fitted, color='red', s=150, zorder=5, label='Romania')
ax.annotate('Romania', (ro_actual, ro_fitted),
            textcoords='offset points', xytext=(8, -12), fontsize=9, color='red')
for i, row in df.iterrows():
    ax.annotate(row['tara'], (Y.iloc[i], y_fitted.iloc[i]),
                fontsize=6, textcoords='offset points', xytext=(3, 2), alpha=0.6)
ax.set_xlabel('Rata phishing actuala (%)')
ax.set_ylabel('Rata phishing prezisa (%)')
ax.set_title(f'Regresie multipla — Valori actuale vs. prezise\n{TITLU_FIRMA}', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('grafic_regresie_ols.png', dpi=150, bbox_inches='tight')
plt.show()

from __future__ import print_function, division
import pandas as pd 
import numpy as np

from gensim import corpora, models, similarities, matutils

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics.pairwise as smp

magic_dataset = "app/data/MagicCards_7_25.csv"
magic_cards = pd.read_csv(magic_dataset, low_memory=False)
magic_cards = magic_cards.drop(magic_cards.columns[0], axis=1)

# Load Combined_DF
# Takes 1:36 to run
# combined_df = pd.read_csv("app/data/Combined_DF")
# combined_df = combined_df.drop(combined_df.columns[0], axis=1)

# Create combined df manually. 
# Takes 50sec to rub
magic_cards_fill = magic_cards.fillna(" ")
magic_cards_fill['combined_text'] = magic_cards_fill['name']+" "+magic_cards_fill['type']+" "+magic_cards_fill['colors']+" "+magic_cards_fill["text"]+" "+magic_cards_fill["flavor"]+" "+magic_cards_fill["rarity"]
magic_cards_s = magic_cards_fill['combined_text']
magic_list = magic_cards_s.tolist
magic_list = list(magic_cards_s)
tfidf = TfidfVectorizer(stop_words="english", token_pattern="\\b[a-zA-Z0-9][a-zA-Z0-9]+\\b", min_df=10)
tfidf_vecs = tfidf.fit_transform(magic_list)
tfidif_df = pd.DataFrame(tfidf_vecs.todense(), columns=tfidf.get_feature_names())
new_cards = magic_cards
new_cards = new_cards.fillna(0.0)
magic_cards_fill_cut = new_cards.iloc[:,[2,11,21,34]]
new_df = magic_cards_fill_cut.astype(str)
dummied_df = pd.get_dummies(new_df)
combined_df = pd.concat([dummied_df, tfidif_df], axis=1, join_axes=[magic_cards_fill_cut.index])

model = NearestNeighbors(n_neighbors=100,n_jobs=-1)
knn = model.fit(combined_df)

set_dict = {
        #Number
        "[3ED]" : "3rd-Edition","[4ED]" : "4th-Edition","[5ED]" : "5th-Edition","[6ED]" : "6th-Edition", 
        "[7ED]" : "7th-Edition","[8ED]" : "8th-Edition","[9ED]" : "9th-Edition","[10E]" : "10th-Edition",
        "[M10]" : "2010-Core-Set","[M11]" : "2011-Core-Set","[M12]" : "2012-Core-Set","[M13]" : "2013-Core-Set", 
        "[M14]" : "2014-Core-Set","[M15]" : "2015-Core-Set",
        #A
        "[AER]" : "Aether-Revolt","AKH" : "Amonkhet","[ARB]" : "Alara-Reborn","[ALL]" : "Alliances","[LEA]" : "Alpha",
        "[ATH]":"Anthologies","[ATQ]" : "Antiquities","[APC]" : "Apocalypse","[ARN]" : "Arabian-Nights",
        "[ARC]" : "Archenemy","[AVR]" : "Avacyn-Restored",
        #B
        "[BFZ]" : "Battle-for-Zendikar","[BRB]" : "Battle-Royale","[BTD]" : "Beatdown","[LEB]" : "Beta",
        "[BOK]" : "Betrayers-of-Kamigawa","[BNG]" : "Born-of-the-Gods", 
        #C
        "[CHK]" : "Champions-of-Kamigawa","[CHR]" : "Chronicles","[CSP]":"Coldsnap","[CED]" : "Collectors'-Edition",
        "[CMD]" : "Commander","[C13]" : "Commander-2013","[C14]" : "Commander-2014","[C15]" : "Commander-2015", 
        "[C16]" : "Commander-2016","[CM1]" : "Commander's-Arsenal","[CON]" : "Conflux","[CNS]" : "Conspiracy",
        "[CN2]" : "Conspiracy-Take-the-Crown",
        #D
        "[DKA]" : "Dark-Ascension","[DST]" : "Darksteel", "[DIS]" : "Dissension", "[DGM]" : "Dragons-Maze",
        "[DTK]" : "Dragons-of-Tarkir",
        #E
        "[EMN]" : "Eldritch-Moon","[EMA]" : "Eternal-Masters","[EVE]" : "Eventide","[EXO]" : "Exodus",
        #F
        "[FEM]" : "Fallen-Empires","[FRF]" : "Fate-Reforged","[5DN]" : "Fifth-Dawn","[FUT]" : "Future-Sight",
        #G-I
        "[GTC]" : "Gatecrash","[GPT]" : "Guildpact","[HML]" : "Homelands","[HOU]" : "Hour-of-Devastation",
        "[ICE]" : "Ice-Age","[ISD]" : "Innistrad","[INV]" : "Invasion",
        #J-L
        "[JOU]" : "Journey-into-Nyx","[JUD]" : "Judgment","[KLD]" : "Kaladesh","[KTK]" : "Khans-of-Tarkir",
        "[LEG]" : "Legends","[LGN]" : "Legions","[LRW]" : "Lorwyn",
        #M
        "[ORI]" : "Magic-Origins","[MMQ]" : "Mercadian-Masques","[MIR]" : "Mirage","[MRD]" : "Mirrodin",
        "[MBS]" : "Mirrodin-Besieged","[MMA]" : "Modern-Masters","[MM2]" : "Modern-Masters-2015","[MOR]" : "Morningtide",
        #N-P
        "[NEM]" : "Nemesis", "[NPH]" : "New-Phyrexia","[OGW]" : "Oath-of-the-Gatewatch", "[ODY]" : "Odyssey",
        "[ONS]" : "Onslaught","[PLC]" : "Planar-Chaos","[HOP]" : "Plancechase","[PC2]" : "Planechase-2012",
        "[PLS]" : "Planeshift","[POR]" : "Portal","[ME2]" : "Portal-II","[PCY]" : "Prophecy",
        #Q-S
        "[RAV]" : "Ravnica","[RTR]" : "Return-to-Ravnica","[ROE]" : "Rise-of-the-Eldrazi", "[SOK]" : "Saviors-of-Kamigawa",
        "[SOM]" : "Scars-of-Mirrodin","[SCG]" : "Scourge","[SHM]" : "Shadowmoor","[SOI]" : "Shadows-Over-Innistrad",
        "[ALA]" : "Shards-of-Alara", "[STH]" : "Stronghold",
        #T-V
        "[TMP]" : "Tempest","[DRK]" : "The-Dark","[THS]" : "Theros","[TSP]" : "Time-Spiral","[TOR]" : "Torment",
        "[UGL]" : "Unglued","[UNH]" : "Unhinged","[2ED]" : "Unlimited","[USG]" : "Urzas-Saga","[ULG]" : "Urzas-Legacy",
        "[UDS]" : "Urzas-Destiny","[VIS]" : "Visions",
        #W-Z
        "[WTH]" : "Weatherlight","[WWK]" : "Worldwake", "[ZEN]" : "Zendikar"

      
}

def checkset(set_list):
    #filter sets until i get one in dict
    for i in range(len(set_list)):
        set_name1 = set_list[i]
        set_name2 = "["+set_name1+"]"
        set_name2 = set_name2.replace(" ","")
        try:
            set_name2 = set_dict[set_name2]
            break
        except:
            pass         
    return set_name2

def predict(name):

    card_index = (magic_cards[magic_cards['name']==name]).index.tolist()
    card_index = card_index[0]

    distances, indices = knn.kneighbors(combined_df.iloc[card_index,:])

    index = indices[0]
    distance = distances[0]

    nearest_list = []
    i=0
    list_dict = []
    for k in range(0,len(index)):
        card_name = magic_cards.iloc[index[k],16]
        if card_name not in nearest_list:
            if i<11:
                card_dict = {}
                nearest_list.append(card_name)
                if i > 0:
                    card_dict['card'] = (str(i) + ".")
                else:
                    card_dict['card'] = ("Original:")
                card_dict['card_name'] = card_name
                card_dict['distance_away'] = distance[k]
                card_dict['cmc'] = str(magic_cards.iloc[index[k],2])
                card_dict['power_toughness'] = str(magic_cards.iloc[index[k],21])+"/"+str(magic_cards.iloc[index[k],34])
                card_dict['cost'] = str(magic_cards.iloc[index[k],12])
                card_dict['type'] = str(magic_cards.iloc[index[k],35])
                card_dict['sets'] = str(magic_cards.iloc[index[k],22])
                card_dict['text'] = str(magic_cards.iloc[index[k],32])
                card_dict['flavor'] = str(magic_cards.iloc[index[k],5])
                
                #setting up link
                card_name2 = card_name.replace(" ", "-").replace(",","").replace("'","").replace(":","")
                set_name = magic_cards.iloc[index[k],22]
                set_list = set_name.replace("[","").replace("]","").split(",")
                #filter sets until i get one in dict
                set_name2 = checkset(set_list)
                card_dict['link'] = "http://www.cardkingdom.com/mtg/"+str(set_name2)+"/"+card_name2
                
                i+=1
                list_dict.append(card_dict)
            else:
                continue
        else:
            continue

    return list_dict



from __future__ import print_function, division
import pandas as pd 
import numpy as np

from gensim import corpora, models, similarities, matutils

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics.pairwise as smp

magic_cards = pd.read_csv("app/data/Magic_Pandas_DF.csv")
magic_cards = magic_cards.drop(magic_cards.columns[0], axis=1)

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

dummied_df = pd.get_dummies(magic_cards_fill_cut)

combined_df = pd.concat([dummied_df, tfidif_df], axis=1, join_axes=[magic_cards_fill_cut.index])

model = NearestNeighbors(n_neighbors=50,n_jobs=-1)
knn = model.fit(combined_df)

set_dict = {
    "[10E]" : "Tenth-Edition", "[2ED]" : "Unlimited", "[3ED]" : "Revised", "[4ED]" : "Fourth-Edition",
    "[5DN]" : "Fifth-Dawn", "[5ED]" : "Fifth-Edition", "[6ED]" : "Sixth-Edition", "[7ED]" : "Seventh-Edition",
    "[8ED]" : "Eighth-Edition", "[9ED]" : "Ninth-Edition", "[ALA]" : "Shards-of-Alara", "[ALL]" : "Alliances",
    "[APC]" : "Apocalypse", "[ARB]" : "Alara-Reborn", "[ARC]" : "Archenemy", "[ARN]" : "Arabian-Nights",
    "[ATQ]" : "Aniquities", "[AVR]" : "Avacyn-Restored", "[BFZ]" : "Battle-for-Zendikar", 
    "[BNG]" : "Born-of-the-Gods", "[BOK]" : "Betrayers-of-Kamigawa", "[BRB]" : "Battle-Royale", "[BTB]" : "Beatdown",              "[C13]" : "Commander 2013",
    "[C14]" : "Commander-2014", "[C15]" : "Commander-2015", "[CED]" : "Collectors'-Edition",
    "[CEI]" : "International-Edition", "[CHK]" : "Champions-of-Kamigawa", "[CHR]" : "Chronicles",
    "[CM1]" : "Commander's-Arsenal", "[CMD]" : "Commander", "[C13]" : "Commander-2013",
    "[DD2]" : "Duel Decks: Jace vs. Chandra", "[DD3_DVD]" : "Duel Decks: Divine vs. Demonic", 
    "[DD3_EVG]" : "Duel Decks: Elves vs. Goblins", "[DD3_GVL]" : "Duel Decks: Garruk vs. Liliana",
    "[DD3_JVC]" : "Duel Decks: Jace vs. Chandra", "[DDC]" : "Duel Decks: Divine vs. Demonic",
    "[DDD]" : "Duel Decks: Garruk vs. Liliana", "[DDF]" : "Duel Decks: Elspeth vs. Tezzeret",
    "[DDG]" : "Duel Decks: Knights vs. Dragons", "[DDH]" : "Duel Deck: Ajani vs. Nicol Bolas",
    "[DDI]" : "Duel Decks: Venser vs. Koth", "[DDJ]" : "Duel Decks: Izzet vs. Golgari", 
    "[DDK]" : "Duel Decks: Sorin vs. Tibalt", "[DDL]" : "Duel Decks: Heroes vs. Monsters",
    "[DDO]" : "Duel Decks: Elspeth vs. Kiora", "[DKA]" : "Dark-Ascension", "[FVD]" : "From-the-Vault:-Dragons",
    "[DRK]" : "The-Dark", "[DTK]" : "Dragons-of-Tarkir", "[EVE]" : "Eventide", "[EVG]" : "Duel Decks: Elves vs. Goblins",
    "[EXO]" : "Exodus", "[EXP]" : "Zendikar-Expeditions", "[FEM]" : "Fallen-Empires", "[FRF]" : "Fate-Reforged",
    "[FUT]" : "Future-Sight", "[HML]" : "Homelands", "[HOP]" : "Plancechase", "[ICE]" : "Ice-Age", "[INV]" : "Invasion",
    "[ISD]" : "Innistrad", "[JOU]" : "Journey-into-Nyx", "[KTK]" : "Khans-of-Tarkir", "[LEA]" : "Alpha", 
    "[LEB]" : "Beta", "[LEG]" : "Legends", "[LGN]" : "Legions", "[LRW]" : "Lorwyn", "[M10]" : "Magic-2010", 
    "[M11]" : "Magic-2011", "[M12]" : "Magic-2012", "[M13]" : "Magic-2013", "[M14]" : "Magic-2014", 
    "[M15]" : "Magic-2015", "[MM2]" : "Modern-Masters-2015", "[MMA]" : "Modern-Masters", "[MMQ]" : "Mercadian-Masques",
    "[MRD]" : "Mirrodin", "[NPH]" : "New-Phyrexia", "[ONS]" : "Onslaught", "[ORI]" : "Magic-Origins", 
    "[PC2]" : "Planechase-2012", "[PCY]" : "Prophecy", "[MIR]" : "Mirage", "[MBS]" : "Mirrodin-Besieged",
    "[CN2]" : "Conspiracy-Take-the-Crown", "[THS]" : "Theros", "[GTC]" : "Gatecrash", "[TSP]" : "Time-Spiral",
    "[OGW]" : "Oath-of-the-Gatewatch", "[SOI]" : "Shadows-Over-Innistrad", "[EMN]" : "Eldritch-Moon", 
    "[KLD]" : "Kaladesh"
}

def predict(name):

    card_index = (magic_cards_fill[magic_cards_fill['name']==name]).index.tolist()
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
            if i<6:
                card_dict = {}
                nearest_list.append(card_name)
                card_dict['card'] = (str(i) + ".")
                card_dict['card_name'] = card_name
                card_dict['distance_away'] = distance[k]
                card_dict['cmc'] = str(magic_cards.iloc[index[k],2])
                card_dict['power_toughness'] = str(magic_cards.iloc[index[k],21])+"/"+str(magic_cards.iloc[index[k],21])
                card_dict['cost'] = str(magic_cards.iloc[index[k],12])
                card_dict['type'] = str(magic_cards.iloc[index[k],35])
                card_dict['sets'] = str(magic_cards.iloc[index[k],22])
                card_dict['text'] = str(magic_cards.iloc[index[k],32])
                card_dict['flavor'] = str(magic_cards.iloc[index[k],5])
                card_name2 = card_name.replace(" ", "-").replace(",","")
                
                #editing set name to get it into cardkingdom form
                set_name = magic_cards.iloc[index[k],22]
                set_name = set_name.replace("[","").replace("]","").split(",")[0]
                set_name = "["+set_name+"]"
                try:
                    set_name2 = set_dict[set_name]
                except:
                    set_name2 = set_name
                card_dict['link'] = "http://www.cardkingdom.com/mtg/"+str(set_name2)+"/"+card_name2
                
                #print checkpoints
                # print (str(i) + ".")
                # print (card_name)
                # print ("Distance Away: "+str(distance[k]))
                # print ("CMC: "+str(magic_cards.iloc[index[k],2])+" Power/Toughness: "+str(magic_cards.iloc[index[k],21])+"/"+str(magic_cards.iloc[index[k],21]))
                # print ("Cost: "+str(magic_cards.iloc[index[k],12]))
                # print ("Type: "+str(magic_cards.iloc[index[k],35]))
                # print ("Sets: "+str(magic_cards.iloc[index[k],22]))
                # print ("Text: "+str(magic_cards.iloc[index[k],32]))
                # print ("Flavor: "+str(magic_cards.iloc[index[k],5]))
                # print ("Link: ")+"http://www.cardkingdom.com/mtg/"+str(set_name)+"/"+card_name2
                # print ("------------")
                
                i+=1
                list_dict.append(card_dict)
            else:
                continue
        else:
            continue

    return list_dict



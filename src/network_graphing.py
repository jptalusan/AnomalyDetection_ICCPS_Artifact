import os
import random
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

def graph_builder(inrix_df, Precision=4):
    inrix_df = inrix_df.copy()
    inrix_df['Beg_']=inrix_df['Beg'].apply(lambda row: (round(row.coords[0][0],Precision), round(row.coords[0][1],Precision)))
    inrix_df['End_']=inrix_df['End'].apply(lambda row: (round(row.coords[0][0],Precision), round(row.coords[0][1],Precision)))
    inrix_df['Center_']=inrix_df['center'].apply(lambda row: (round(row.coords[0][0],Precision),round(row.coords[0][1],Precision)))

    Transfered_Features=['XDSegID','FRC','Miles']
    G = nx.from_pandas_edgelist(inrix_df,source ='Beg_', target ='End_', edge_attr=Transfered_Features, create_using=nx.MultiDiGraph())
    print('Number of the Nodes in the Graph:',len(list(G.nodes())))
    print('Number of the Edges in the Graph:',len(list(G.edges())))
    return G

def line_graph(G, inrix_df):
    L = nx.empty_graph(0, create_using=G.__class__)
    source_list=[]
    target_list=[]
    for from_node in G.edges(None, keys=True, data=True):
        L.add_node(from_node[0:3], XDSegID=from_node[3]['XDSegID'], 
                   FRC = from_node[3]['FRC'], Miles=from_node[3]['Miles'])
        
        for to_node in G.edges(from_node[1], keys=True, data=True):
            L.add_node(to_node[0:3], XDSegID=to_node[3]['XDSegID'], FRC=to_node[3]['FRC'], Miles=from_node[3]['Miles'])
            L.add_edge(from_node[0:3], to_node[0:3])
            source_list.append(from_node[3]['XDSegID'])
            target_list.append(to_node[3]['XDSegID'])
            
    DF_adj = pd.DataFrame({'source':source_list,'target':target_list})
    DF_adj = inrix_info_incorporation_graph(L, inrix_df, DF_adj)

    L_XDSegID=nx.from_pandas_edgelist(DF_adj, source ='source', target ='target', create_using=nx.MultiDiGraph())
    print('Number of the Nodes in the Line_Graph_XDSegID:',len(list(L_XDSegID.nodes())))
    print('Number of the Edges in the Line_Graph_XDSegID:',len(list(L_XDSegID.edges())))
    inrix_df=inrix_df.copy()
    inrix_df['Center_']=inrix_df['center'].apply(lambda row: (row.coords[0][0],row.coords[0][1]))
    XDSegID_Center = dict(zip(inrix_df.XDSegID, inrix_df.Center_))
    for nodes in L_XDSegID.nodes():
        L_XDSegID.nodes[nodes]['center']=XDSegID_Center[nodes]
    
    return DF_adj, L_XDSegID

def inrix_info_incorporation_graph(L, inrix_df, DF_adj):
    for i,row in inrix_df.iterrows():
        if (row['PreviousXD'] is None)  | (np.isnan(row['PreviousXD']) ) ==False:
            if row['PreviousXD'] in inrix_df['XDSegID'].tolist():
                if any((DF_adj['source']==row['PreviousXD']) & (DF_adj['target']==row['XDSegID']))==False:
                    DF_added_Edge=pd.DataFrame({'source':[row['PreviousXD']],'target':[row['XDSegID']]})
                    DF_adj=DF_adj.append(DF_added_Edge, ignore_index=True)

        if (row['NextXDSegI'] is None)  | (np.isnan(row['NextXDSegI']) ) ==False:
            if row['NextXDSegI'] in inrix_df['XDSegID'].tolist():
                if any((DF_adj['source']==row['XDSegID']) & (DF_adj['target']==row['NextXDSegI']))==False:
                    DF_added_Edge=pd.DataFrame({'source':[row['XDSegID']],'target':[row['NextXDSegI']]})
                    DF_adj=DF_adj.append(DF_added_Edge, ignore_index=True)
        
    return DF_adj

def prepare_graph(inrix_df, precision=4):
    print("Building graph process is started.")
    
    G = graph_builder(inrix_df, precision)
    DF_adj, L_XDSegID = line_graph(G, inrix_df)
    
    return DF_adj, L_XDSegID, G
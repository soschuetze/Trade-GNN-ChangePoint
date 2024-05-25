# years = range(1962,2019)

# train_years = [2005, 1969, 2002, 1997, 1993, 1982, 2001, 2000, 1962, 1985, 1978, 2016, 1986, 1987, 1989, 1971, 2013, 1996, 1995, 1967, 2017, 1974, 1990, 1977, 1980, 2014, 1965, 1984, 2006, 1973, 1968, 1981, 1970, 1991]
# val_years = [1975, 1983, 2009, 1966, 1999, 1988, 2007, 1979, 1972, 2015, 2003]
# test_years = [1963, 1964, 1976, 1992, 1994, 1998, 2004, 2008, 2010, 2011, 2012, 2018]

# train_graphs = []
# val_graphs = []
# test_graphs = []
# i = 0

# for year in tqdm(years):
#     print(str(year), end='\r')
    
#     trade = TradeNetwork(year = year)
#     trade.prepare_features()
#     trade.prepare_network()
#     trade.graph_create(node_features = ['prev_gdp_per_cap_growth', 'current_gdp_per_cap_growth',
#     'resource_0', 'resource_1', 'resource_2', 'resource_3', 'resource_4', 'resource_5', 'resource_6', 'resource_7',
#        'resource_8', 'resource_9'],
#         node_labels = 'future_gdp_per_cap_growth')
    
#     if(year in val_years):
#         val_graphs.append(trade.pyg_graph)
#     elif(year in test_years):
#         test_graphs.append(trade.pyg_graph)
#     else: 
#         train_graphs.append(trade.pyg_graph)
        
#     trade.features["year"] = year
    
#     if(i == 0):
#         trade_df = trade.features
#     else: 
#         trade_df = pd.concat([trade_df, trade.features])
        
#     i = i+1
#     print(trade.node_attributes.size())
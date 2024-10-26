import torch
import pickle as pkl
from CreateFeatures import CreateFeatures
from functions import dist_labels_to_changepoint_labels
from sample import sample_pairs
import json

def add_features(years, graphs, feat_dict, dim):

    all_nodes = ['ABW', 'AFG', 'AGO', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM',
       'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BFA', 'BGD',
       'BGR', 'BHR', 'BHS', 'BIH', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA',
       'BRB', 'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CHE', 'CHL', 'CHN',
       'CIV', 'CMR', 'COD', 'COG', 'COL', 'COM', 'CPV', 'CRI', 'CUB',
       'CUW', 'CYM', 'CYP', 'CZE', 'DEU', 'DMA', 'DNK', 'DOM', 'DZA',
       'ECU', 'EGY', 'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FRA', 'FSM',
       'GAB', 'GBR', 'GEO', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'GRC',
       'GRD', 'GRL', 'GTM', 'GUM', 'GUY', 'HKG', 'HND', 'HRV', 'HTI',
       'HUN', 'IDN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA',
       'JAM', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KNA', 'KOR',
       'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LKA', 'LSO', 'LTU',
       'LUX', 'LVA', 'MAC', 'MAR', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL',
       'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MNP', 'MOZ', 'MRT',
       'MUS', 'MWI', 'MYS', 'NAM', 'NER', 'NGA', 'NIC', 'NLD', 'NOR',
       'NPL', 'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW',
       'PNG', 'POL', 'PRT', 'PRY', 'PSE', 'PYF', 'QAT', 'ROU', 'RUS',
       'RWA', 'SAU', 'SDN', 'SEN', 'SGP', 'SLB', 'SLE', 'SLV', 'SMR',
       'SRB', 'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SXM',
       'SYC', 'SYR', 'TCD', 'TGO', 'THA', 'TJK', 'TKM', 'TLS', 'TON',
       'TTO', 'TUN', 'TUR', 'TUV', 'TZA', 'UGA', 'UKR', 'URY', 'USA',
       'UZB', 'VCT', 'VEN', 'VNM', 'VUT', 'WSM', 'YEM', 'ZAF', 'ZMB',
       'ZWE']
    
    zeros = torch.zeros(dim)

    for i in range(len(years)):
        new_x = torch.empty(0, dim)
        year = years[i]

        feat_dict_year = feat_dict[year].combined_features

        for j, country in enumerate(all_nodes):
            if j == 0:
                new_x = torch.stack([zeros])

            elif country in feat_dict_year["country_code"].values:
                tensor_before = graphs[i].x[j]
                country_row = feat_dict_year[feat_dict_year["country_code"] == country]
                country_row = country_row.drop(columns = ["country_code", "current_gdp_growth"])
                row_values = country_row.values.tolist()
                row_tensor = torch.tensor(row_values)[0]
                combined_values = torch.cat((tensor_before, row_tensor))

                new_x = torch.cat((new_x, combined_values.unsqueeze(0)), dim=0)

            else:
                new_x = torch.cat((new_x, zeros.unsqueeze(0)), dim=0)

        graphs[i].x = new_x

    return graphs

def get_window_graph_pairs():
    
    years = range(1962,2019)

    with open("../pygcn/all_graphs.pkl", "rb") as f:         
        all_graphs = pkl.load(f)

    with open("../../feature_dicts/mis_norm.pkl", "rb") as f:
        feat_dict = pkl.load(f)

    all_graphs = add_features(years, all_graphs, feat_dict, 27)

    train_graphs = all_graphs[:34]
    val_graphs = all_graphs[34:45]
    test_graphs = all_graphs[45:]
    
    crisis_years = [1962, 1967, 1973, 1978, 1981, 1989, 1993, 1996, 2002, 2007, 2012, 2014, 2016]
    phases = []
    p = -1
    for i in range(1962,2019):
        if i in crisis_years:
            p += 1
        phases.append(p)

    labels = dist_labels_to_changepoint_labels(phases)

    time_test = [t-2007 for t in crisis_years if t>=2007]
    with open("../../results/test_window/window-graphs.p", "wb") as f:         
        pkl.dump(test_graphs, f)
    with open("../../results/test_window/window-labels.p", "wb") as f:         
        pkl.dump(labels[45:], f)
    with open("../../results/test_window/window-time.json", "w") as f:         
        json.dump(time_test, f)

    graph_pairs_train = sample_pairs(train_graphs,labels[:34])
    graph_pairs_val = sample_pairs(val_graphs,labels[34:45])
    graph_pairs_test = sample_pairs(test_graphs,labels[45:])

    for g in graph_pairs_train:
        if torch.is_tensor(g[0].x):
            g[0].x = g[0].x.numpy()
        if torch.is_tensor(g[1].x):
            g[1].x = g[1].x.numpy()

    for g in graph_pairs_val:
        if torch.is_tensor(g[0].x):
            g[0].x = g[0].x.numpy()
        if torch.is_tensor(g[1].x):
            g[1].x = g[1].x.numpy()

    for g in graph_pairs_test:
        if torch.is_tensor(g[0].x):
            g[0].x = g[0].x.numpy()
        if torch.is_tensor(g[1].x):
            g[1].x = g[1].x.numpy()

    with open("../../window/graph_pairs/graph_pairs_train.pkl", "wb") as f:         
        pkl.dump(graph_pairs_train, f)
    with open("../../window/graph_pairs/graph_pairs_test.pkl", "wb") as f:         
        pkl.dump(graph_pairs_test, f)
    with open("../../window/graph_pairs/graph_pairs_val.pkl", "wb") as f:         
        pkl.dump(graph_pairs_val, f)

def main():
    
    get_window_graph_pairs()

if __name__ == '__main__':
    main()
# from tqdm import tqdm
# from src.utils.TradeNetwork import TradeNetwork
# import pandas as pd
# import pickle as pkl
# import numpy as np

# years = range(1962,2019)

# graphs = []
# # train_years = [2005, 1969, 2002, 1997, 1993, 1982, 2001, 2000, 1962, 1985, 1978, 2016, 1986, 1987, 1989, 1971, 2013, 1996, 1995, 1967, 2017, 1974, 1990, 1977, 1980, 2014, 1965, 1984, 2006, 1973, 1968, 1981, 1970, 1991]
# # val_years = [1975, 1983, 2009, 1966, 1999, 1988, 2007, 1979, 1972, 2015, 2003]
# # test_years = [1963, 1964, 1976, 1992, 1994, 1998, 2004, 2008, 2010, 2011, 2012, 2018]

# # train_graphs = []
# # val_graphs = []
# # test_graphs = []
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
    
#     graphs.append(trade.pyg_graph)
        
#     trade.features["year"] = year
    
#     if(i == 0):
#         trade_df = trade.features
#     else: 
#         trade_df = pd.concat([trade_df, trade.features])
        
#     i = i+1
#     print(trade.node_attributes.size())

# with open('graphs/all_graphs.pkl', 'wb') as f:
#     pkl.dump(graphs, f)



# for time, data in zip(cp_times, merge_data):
#     for j, i in enumerate(data):
#         edge_index = i.edge_index.to(torch.int64)
#         networkx_graph = to_networkx(i)
#         adjacency = nx.to_scipy_sparse_array(networkx_graph, format='csr')
                
#         x = np.diag(degree_matrix(adjacency).todense(), k=0).reshape(-1,1)
#         data[j].x = x

#     flattened_train, flattened_test, flattened_val = create_synthetic_pairs(data, time)

#     positive_samples = [item for item in flattened_train if item[2] == 1]
#     negative_samples = [item for item in flattened_train if item[2] == 0]

#     print(len(positive_samples))
#     print(len(negative_samples))

#     # Calculate the difference in count
#     diff = len(negative_samples) - len(positive_samples)

#     # Upsample positive samples
#     if diff > 0:
#         positive_samples_upsampled = positive_samples * (diff // len(positive_samples)) + random.sample(positive_samples, diff % len(positive_samples))
#         balanced_data = negative_samples + positive_samples + positive_samples_upsampled
#     else:
#         balanced_data = flattened_train

#     random.shuffle(balanced_data)

#     run_model(balanced_data, flattened_val)




    # #input_dim = len(training_data_pairs.dataset[0][0].x)
    # input_dim = 102780

    # embedding = GCN(input_dim=input_dim, type="identity", hidden_dim=16, layers=3, dropout=0.1)

    # model = GraphSiamese(embedding, "euclidean", "topk", "bce", 30, nlinear=2,
    #                      nhidden=16, dropout=0.1, features='identity')
    
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # logging = {'train_loss': [], 'train_acc': [], 'train_recall': [], 'train_precision': [], 'train_auc': [], 'train_f1': [],
    #                'valid_loss': [], 'valid_acc': [], 'valid_precision': [], 'valid_recall': [], 'valid_auc': [], 'valid_f1': []}
    # best_f1, best_weights, best_loss = 0., None, np.Inf
    # final_metrics = {'loss' : [0.0, 0., 0.], 'accuracy': [0., 0., 0.], 'recall': [0., 0., 0.], 'precision': [0., 0., 0.],
    #                  'auc': [0., 0., 0.], 'f1': [0., 0., 0.]}
    # patience = 10
    # patience_counter = 0

    # loss_fn = torch.nn.BCELoss(reduction='none')

    # for epoch in range(20):

    #     # training updates
    #     train_loss, train_acc, train_precision, train_recall, train_auc, train_f1 = [], [], [], [], [], []

    #     model.train()

    #     # minibatch loop
    #     for (graph1, graph2, labels) in training_data_pairs:

    #         predictions = model(graph1, graph2)

    #         predictions = torch.sigmoid(predictions) # predictions between 0 and 1
            
    #         loss = loss_fn(predictions.squeeze(), labels.squeeze())

    #         # balanced accuracy score instead of plain accuracy
    #         accuracy = torch.tensor(np.array((predictions.squeeze().cpu().detach() > 0.5) == labels.cpu(),
    #                                          dtype=float).mean().item()).unsqueeze(dim=0)
    #         recall = torch.tensor(
    #             [recall_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float(), zero_division=0.)])
    #         precision = torch.tensor(
    #             [precision_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float(), zero_division=0.)])

    #         if (labels != 0).all() or (labels != 1).all():
    #             print("No positive or negative labels in this minibatch of the training set")
    #             auc = torch.zeros_like(accuracy)
    #         else:
    #             auc = torch.tensor([roc_auc_score(labels.cpu(), predictions.squeeze().cpu().detach())])
    #         f1 = torch.tensor([f1_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float())])

    #         train_loss.append(loss), train_acc.append(accuracy), train_recall.append(recall), train_precision.append(precision), \
    #         train_auc.append(auc), train_f1.append(f1)

    #         optimizer.zero_grad()
    #         loss.mean().backward()
    #         optimizer.step()


    #     logging['train_loss'].append(torch.cat(train_loss).mean().item())
    #     logging['train_acc'].append(torch.cat(train_acc).mean().item())
    #     logging['train_recall'].append(torch.cat(train_recall).mean().item())
    #     logging['train_precision'].append(torch.cat(train_precision).mean().item())
    #     logging['train_auc'].append(torch.cat(train_auc).mean().item())
    #     logging['train_f1'].append(torch.cat(train_f1).mean().item())

    #     scheduler.step()
    #     validation_proportion = 0.5
    #     # test step
    #     if validation_proportion > 0.0:

    #         # validation step
    #         model.eval()

    #         valid_loss, valid_acc, valid_recall, valid_precision, valid_auc, valid_f1 = [], [], [], [], [], []
    #         with torch.no_grad():
    #             for (graph1, graph2, labels) in validation_data_pairs:

    #                 predictions = model(graph1, graph2)

    #                 predictions = torch.sigmoid(predictions)

    #                 loss = loss_fn(predictions.squeeze(), labels.squeeze())

    #                 recall = torch.tensor(
    #                     [recall_score(labels.float(), (predictions.squeeze().detach().cpu() > 0.5).float(), zero_division=0.)])
    #                 precision = torch.tensor(
    #                     [precision_score(labels.float(), (predictions.squeeze().detach().cpu() > 0.5).float(), zero_division=0.)])
    #                 accuracy = torch.tensor(np.array((predictions.squeeze().detach().cpu() > 0.5).float() == labels.float(),
    #                                                  dtype=float).mean().item()).unsqueeze(dim=0)

    #                 if (labels != 0).all() or (labels != 1).all():
    #                     #print(labels)
    #                     print("No positive or negative labels in this minibatch of the validation set")
    #                     auc = torch.zeros_like(accuracy)
    #                 else:
    #                     auc = torch.tensor([roc_auc_score(labels.float(), predictions.squeeze().detach().cpu())])
    #                 f1 = torch.tensor([f1_score(labels.cpu(), (predictions.squeeze().cpu().detach() > 0.5).float())])

    #                 valid_loss.append(loss), valid_acc.append(accuracy), valid_recall.append(
    #                     recall), valid_precision.append(
    #                     precision), valid_auc.append(auc), valid_f1.append(f1)

    #             logging['valid_loss'].append(torch.cat(valid_loss).mean().item())
    #             logging['valid_acc'].append(torch.cat(valid_acc).mean().item())
    #             logging['valid_recall'].append(torch.cat(valid_recall).mean().item())
    #             logging['valid_precision'].append(torch.cat(valid_precision).mean().item())
    #             logging['valid_auc'].append(torch.cat(valid_auc).mean().item())
    #             logging['valid_f1'].append(torch.cat(valid_f1).mean().item())

    #         # save best weights
    #         #if logging['valid_f1'][-1] > best_f1 and epoch > 0:
    #         if logging['valid_loss'][-1] < best_loss and epoch > 0:
    #             best_loss = logging['valid_loss'][-1]
    #             #best_f1 = logging['valid_f1'][-1]
    #             final_metrics['loss'][:2] = [logging['train_loss'][-1], logging['valid_loss'][-1]]
    #             final_metrics['accuracy'][:2] = [logging['train_acc'][-1], logging['valid_acc'][-1]]
    #             final_metrics['recall'][:2] = [logging['train_recall'][-1], logging['valid_recall'][-1]]
    #             final_metrics['precision'][:2] = [logging['train_precision'][-1], logging['valid_precision'][-1]]
    #             final_metrics['auc'][:2] = [logging['train_auc'][-1], logging['valid_auc'][-1]]
    #             final_metrics['f1'][:2] = [logging['train_f1'][-1], logging['valid_f1'][-1]]
    #             best_weights = model.state_dict()
    #             patience_counter = 0
    #         else:
    #             patience_counter += 1


    #     if patience == patience_counter:
    #         break

    #     train_f1, train_loss = logging['train_f1'][-1], logging['train_loss'][-1]

    #     if validation_proportion > 0.0:
    #         valid_acc, valid_loss, valid_f1 = logging['valid_acc'][-1], logging['valid_loss'][-1], logging['valid_f1'][-1]
    #         print("Epoch, Training loss, f1, Valid loss, f1 :", epoch, train_loss, train_f1, valid_loss,
    #             valid_f1)
    #         print("Patience counter : ", patience_counter)

    #     else:
    #         print("Epoch, Training loss, accuracy :", epoch, train_loss, train_f1)
    #     #print("CPU Memory usage :", process.memory_info().rss)
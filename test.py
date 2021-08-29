test_data = pd.read_csv(test_dataset_path)

__ids2intents, __ids2slots, vectorized_slots_test, vectorized_intents_test, filepaths_test = dataset_processor.process_data(test_data)

save_obj(vectorized_slots_test, 'vectorized_slots_test')
save_obj(vectorized_intents_test, 'vectorized_intents_test') 

test_generator = DataGenerator([filepaths_test, vectorized_intents_test, vectorized_slots_test], 
                                     [n_classes, n_slots], batch_size = batch_size, vis = False,
                                     shuffle=False, to_fit=True, augment = False)

data = test_generator.__getitem__(0)
print(data[0].shape)
print(data[1][0].shape)
print(data[1][1].shape)
print(test_generator.__len__())

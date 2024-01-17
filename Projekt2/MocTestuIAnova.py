import random

def cut_data_to_the_same_size(data):
    numbers_of_datas = list(range(7))
    matrix_of_colors = ["Black","White","Blue","Silver","Red","Green","Gold"]
    for i in range(0,7):
            numbers_of_datas[i] = np.count_nonzero(data[2] == matrix_of_colors[i])
    how_big_groups = min(numbers_of_datas)
    for i in range(0,7):
        indeksy = np.where(data[:, 2] == matrix_of_colors[i])
        wanted_data = data[indeksy]
        if how_big_groups < np.shape(wanted_data)[0]:
            indices_for_train =  random.sample(range(0, wanted_data.shape[0]), int(how_big_groups))
            new_data = wanted_data[indices_for_train,:]
        else:
            new_data = wanted_data
        if i == 0:
            returner = new_data
        else:
            np.vstack([returner,new_data])
    return returner
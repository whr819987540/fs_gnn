def get_layer_size(n_feat, n_hidden, n_class, n_layers):
    layer_size = [n_feat]
    layer_size.extend([n_hidden] * (n_layers - 1))
    layer_size.append(n_class)
    return layer_size

# [20, 10, 10, 5]
print(get_layer_size(20,10,5,3))
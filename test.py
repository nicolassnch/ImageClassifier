
train_data = {-1 :["aaa", "bbb", "ccc"],
              1: ["ddd", "eee", "fff"]}

X_train = []
y_train = []

for i in train_data[-1]:
    y_train.append(-1)
    X_train.append(i)

for i in train_data[1]:
    y_train.append(1)
    X_train.append(i)

print(X_train)
print(y_train)

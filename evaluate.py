from dataset import *
from model import *
import matplotlib.pyplot as plt
import pandas as pd

csv = './all_indices_data.csv'


def predict(model, inputs):
    model.eval()
    with torch.no_grad:
        predictions = model(inputs)

    return predictions

# prepare data


def prep_data(model, dataset_test, y_test, start, end, last):
    # prepare test data X
    dataset_test_X = dataset_test[start:end, :]
    test_X_new = dataset_test_X.reshape(
        1, dataset_test_X.shape[0], dataset_test_X.shape[1])
    x_test = torch.tensor(test_X_new, dtype=torch.float32)
    # prepare past and groundtruth
    past_data = y_test[start:end, :]

    dataset_test_y = y_test[end:last, :]
    #scaler1 = MinMaxScaler(feature_range=(0, 1))
    # scaler1.fit(dataset_test_y)

    #xscaler = pickle.load(open("xscaler.pkl",'rb'))
    yscaler = pickle.load(open("yscaler.pkl", 'rb'))

    # predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
    #print(f'ypred shape = {y_pred.shape}')
    y_pred_inv = yscaler.inverse_transform(y_pred)
    y_pred_inv = y_pred_inv.reshape(predict_day, 1)
    y_pred_inv = y_pred_inv[:, 0]

    dataset_test_y_inv = yscaler.inverse_transform(dataset_test_y)
    past_data_inv = yscaler.inverse_transform(past_data)

    return y_pred_inv, dataset_test_y_inv, past_data_inv


def evaluate_prediction(predictions, actual, model_name, start, end):
    predictions = np.squeeze(predictions)
    actual = np.squeeze(actual)
    errors = predictions - actual
    mape = np.mean(np.abs(errors / actual)) * 100
    print('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(mape))

# plot graph


def plotline(history, y_true, y_predict):
    plt.figure(figsize=(20, 4))
    range_history = len(history)
    range_future = list(range(range_history, range_history+len(y_predict)))
    # print(np.array(y_predict))
    plt.plot(np.arange(range_history), np.array(history), label='history')
    plt.plot(range_future, np.array(y_predict), label='prediction')
    plt.plot(range_future, np.array(y_true), label='GroundTruth')
    plt.xlabel("time")
    plt.ylabel("stock price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    load_model = modell(n_features=n_features, model_path='./Adam_2499.pth')
    datasets, y_true = dataprocessing(
        csv=csv, split_date='2018/12/24', mode='test', ticker='^GDAXI')
    for i in range(0, 15, 3):
        start = i
        end = start+previous_days
        last = end + predict_day
        y_pred_inv, dataset_test_y, past_data = prep_data(
            load_model, datasets, y_true, start, end, last)
        evaluate_prediction(y_pred_inv, dataset_test_y, 'LSTM', start, end)
        data_df = pd.DataFrame()
        data_df["true"] = list(dataset_test_y.reshape(predict_day))
        data_df["pred"] = list(y_pred_inv.transpose())
        # print(data_df)
        # print(f'data_test_y:{dataset_test_y},y_pre_inv={y_pred_inv}')
        plotline(past_data, dataset_test_y, y_pred_inv)

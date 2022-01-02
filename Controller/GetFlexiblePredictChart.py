import matplotlib.pyplot as plt
import pandas as pd

filename = 'predict_results_all_range.csv'
df_all = pd.read_csv(filename)


def Draw(df, A_state, B_state):
    plt.figure()
    flagLegend_ok, flagLegend_ng = False, False
    for row in range(len(df)):
        row_val = df.iloc[row]

        temp_A = row_val[2]
        temp_B = row_val[3]
        label = row_val[4]

        plt.xlabel("Machine A Temprature")
        plt.ylabel("Machine B Temprature")
        plt.title(f"Machine A state: {A_state}, Machine B State: {B_state}")

        if label == 'OK':
            color = 'b'
        else:
            color = 'r'
        plt.scatter(temp_A, temp_B, s=2, alpha=1, c=color)

        if not flagLegend_ok:
            plt.scatter(temp_A, temp_B, s=2, alpha=1, c='b', label='OK')
            flagLegend_ok = True

        if not flagLegend_ng:
            plt.scatter(temp_A, temp_B, s=2, alpha=1, c='r', label='NG')
            flagLegend_ng = True
    plt.legend()
    plt.show()


ma, mb = 0, 1
df_1 = df_all[(df_all['State_A'] == ma) & (df_all['State_B'] == mb)]
Draw(df_1, ma, mb)

ma, mb = 1, 0
df_2 = df_all[(df_all['State_A'] == ma) & (df_all['State_B'] == mb)]
Draw(df_2, ma, mb)

import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    C = int(data[3])
    
    index = 4
    D = []
    for _ in range(N):
        row = list(map(int, data[index:index+N]))
        D.append(row)
        index += N
    
    INF = math.inf
    dp_car = [INF] * (N + 1)
    dp_train = [INF] * (N + 1)
    dp_car[1] = 0  # Starting point with car
    
    # Precompute car and train times
    car_times = [[D[j][k] * A for k in range(N)] for j in range(N)]
    train_times = [[D[j][k] * B + C for k in range(N)] for j in range(N)]
    
    # Iterate through all cities to update dp_car and dp_train
    for j in range(1, N + 1):
        if dp_car[j] != INF:
            for k in range(1, N + 1):
                if j != k:
                    time_car = dp_car[j] + car_times[j-1][k-1]
                    if time_car < dp_car[k]:
                        dp_car[k] = time_car
                    time_train = dp_car[j] + train_times[j-1][k-1]
                    if time_train < dp_train[k]:
                        dp_train[k] = time_train
        if dp_train[j] != INF:
            for k in range(1, N + 1):
                if j != k:
                    time_train = dp_train[j] + train_times[j-1][k-1]
                    if time_train < dp_train[k]:
                        dp_train[k] = time_train
    
    answer = min(dp_car[N], dp_train[N])
    print(int(answer))

if __name__ == "__main__":
    main()
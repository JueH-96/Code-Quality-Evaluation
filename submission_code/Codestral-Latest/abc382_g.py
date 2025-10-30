import sys
from math import ceil

def tile_coordinates(x, y, K):
    i = x // K
    j = y // K
    if (i % 2 == j % 2):
        k = y % K
    else:
        k = x % K
    return i, j, k

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def min_moves(K, S_x, S_y, T_x, T_y):
    S_i, S_j, S_k = tile_coordinates(S_x, S_y, K)
    T_i, T_j, T_k = tile_coordinates(T_x, T_y, K)

    if (S_i % 2 == S_j % 2) == (T_i % 2 == T_j % 2):
        return manhattan_distance(S_i, S_j, T_i, T_j) + abs(S_k - T_k)
    else:
        return manhattan_distance(S_i, S_j, T_i, T_j) + abs(S_k - T_k) + 1

def main():
    input = sys.stdin.read
    data = input().split()

    T = int(data[0])
    index = 1
    results = []

    for _ in range(T):
        K = int(data[index])
        S_x = int(data[index + 1])
        S_y = int(data[index + 2])
        T_x = int(data[index + 3])
        T_y = int(data[index + 4])
        index += 5

        result = min_moves(K, S_x, S_y, T_x, T_y)
        results.append(result)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()
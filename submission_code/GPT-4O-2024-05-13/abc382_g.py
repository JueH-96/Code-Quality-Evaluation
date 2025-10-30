# YOUR CODE HERE
import sys
input = sys.stdin.read

def find_tile(x, y, K):
    i = x // K
    j = y // K
    return i, j

def min_moves(K, S_x, S_y, T_x, T_y):
    S_i, S_j = find_tile(S_x, S_y, K)
    T_i, T_j = find_tile(T_x, T_y, K)
    return abs(S_i - T_i) + abs(S_j - T_j)

def main():
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
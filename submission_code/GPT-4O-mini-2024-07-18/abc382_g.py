def find_tile(K, x, y):
    i = int(x // K)
    j = int(y // K)
    
    if (i % 2) == (j % 2):  # same parity
        k = int(y % K)
        return (i, j, k)
    else:  # different parity
        k = int(x % K)
        return (i, j, k)

def min_moves(K, S_x, S_y, T_x, T_y):
    start_tile = find_tile(K, S_x + 0.5, S_y + 0.5)
    target_tile = find_tile(K, T_x + 0.5, T_y + 0.5)
    
    # Calculate the Manhattan distance between the two tiles
    return abs(start_tile[0] - target_tile[0]) + abs(start_tile[1] - target_tile[1]) + abs(start_tile[2] - target_tile[2])

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    T = int(data[0])
    results = []
    
    for i in range(1, T + 1):
        K, S_x, S_y, T_x, T_y = map(int, data[i].split())
        moves = min_moves(K, S_x, S_y, T_x, T_y)
        results.append(moves)
    
    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()
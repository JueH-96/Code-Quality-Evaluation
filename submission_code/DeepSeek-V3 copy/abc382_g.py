def get_tile_coordinates(x, y, K):
    i = x // K
    j = y // K
    if (i % 2 == j % 2):
        k = int(y - j * K)
    else:
        k = int(x - i * K)
    return (i, j, k)

def min_moves(K, S_x, S_y, T_x, T_y):
    s_i, s_j, s_k = get_tile_coordinates(S_x + 0.5, S_y + 0.5, K)
    t_i, t_j, t_k = get_tile_coordinates(T_x + 0.5, T_y + 0.5, K)
    
    # Calculate the differences in i and j
    di = abs(t_i - s_i)
    dj = abs(t_j - s_j)
    
    # If the parities of i and j are the same, the k values are in the same direction
    if (s_i % 2 == s_j % 2) == (t_i % 2 == t_j % 2):
        # The k values are in the same direction
        # The number of moves is the maximum of di and dj
        return max(di, dj)
    else:
        # The k values are in different directions
        # The number of moves is di + dj
        return di + dj

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    index = 1
    for _ in range(T):
        K = int(data[index])
        S_x = int(data[index+1])
        S_y = int(data[index+2])
        T_x = int(data[index+3])
        T_y = int(data[index+4])
        index += 5
        print(min_moves(K, S_x, S_y, T_x, T_y))

if __name__ == "__main__":
    main()
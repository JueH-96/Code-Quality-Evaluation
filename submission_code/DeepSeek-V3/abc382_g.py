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
    
    di = abs(s_i - t_i)
    dj = abs(s_j - t_j)
    
    if di == 0 and dj == 0:
        return abs(s_k - t_k)
    
    if di == 0:
        return dj * K + abs(s_k - t_k)
    
    if dj == 0:
        return di * K + abs(s_k - t_k)
    
    return di * K + dj * K

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
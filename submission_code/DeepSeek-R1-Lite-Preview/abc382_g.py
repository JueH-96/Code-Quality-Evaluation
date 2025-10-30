def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    T = int(data[0])
    index = 1
    for _ in range(T):
        K = int(data[index])
        Sx = int(data[index+1])
        Sy = int(data[index+2])
        Tx = int(data[index+3])
        Ty = int(data[index+4])
        index += 5

        def get_tile(x, y, K):
            i = x // K
            j = y // K
            if (i % 2 == j % 2):
                k = y - j * K
            else:
                k = x - i * K
            return i, j, k

        i_s, j_s, k_s = get_tile(Sx, Sy, K)
        i_t, j_t, k_t = get_tile(Tx, Ty, K)
        
        minimal_moves = abs(i_t - i_s) + abs(j_t - j_s) + abs(k_t - k_s)
        print(minimal_moves)

if __name__ == "__main__":
    main()
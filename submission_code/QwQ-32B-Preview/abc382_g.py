def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        K = int(data[idx])
        S_x = int(data[idx + 1])
        S_y = int(data[idx + 2])
        T_x = int(data[idx + 3])
        T_y = int(data[idx + 4])
        idx += 5
        
        # Function to get tile coordinates
        def get_tile(x, y, K):
            i = x // K
            j = y // K
            if (i % 2 == j % 2):
                # same parity
                k = (y // 1) - (j * K)
            else:
                # different parity
                k = (x // 1) - (i * K)
            return i, j, k
        
        # Get source and target tiles
        i1, j1, k1 = get_tile(S_x + 0.5, S_y + 0.5, K)
        i2, j2, k2 = get_tile(T_x + 0.5, T_y + 0.5, K)
        
        # Calculate differences
        dx = abs(i1 - i2)
        dy = abs(j1 - j2)
        dk = abs(k1 - k2)
        
        # Calculate minimum moves
        distance = dx + dy + max(0, dk - min(dx, dy))
        print(distance)

if __name__ == "__main__":
    main()
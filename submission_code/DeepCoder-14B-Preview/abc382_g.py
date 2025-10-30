def compute_tile(x, y, K):
    i = x // K
    j = y // K
    if (i % 2 == j % 2):
        # Same parity: x is in [i*K, (i+1)*K) and y in [j*K + k, j*K + k + 1)
        # So for same parity, k = (y - j*K)
        k = y - j * K
    else:
        # Different parity: x is in [i*K +k, i*K +k +1) and y is in [j*K, (j+1)*K)
        # So k = x - i*K
        k = x - i * K
    return (i, j, int(k))

def minimal_steps(start, end):
    (i1, j1, k1) = start
    (i2, j2, k2) = end
    di = abs(i1 - i2)
    dj = abs(j1 - j2)
    dk = abs(k1 - k2)
    # The minimal steps are the sum of the differences
    return di + dj + dk

def main():
    import sys
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    for _ in range(T):
        K = int(input[idx])
        Sx = int(input[idx+1])
        Sy = int(input[idx+2])
        Tx = int(input[idx+3])
        Ty = int(input[idx+4])
        idx +=5
        
        # Compute start and end tiles
        start_x = Sx + 0.5
        start_y = Sy + 0.5
        end_x = Tx + 0.5
        end_y = Ty + 0.5
        
        # Compute i, j, k for start
        i1 = int(start_x // K)
        j1 = int(start_y // K)
        if (i1 % 2 == j1 % 2):
            k1 = int(start_y - j1 * K)
        else:
            k1 = int(start_x - i1 * K)
        start_tile = (i1, j1, k1)
        
        # Compute i, j, k for end
        i2 = int(end_x // K)
        j2 = int(end_y // K)
        if (i2 % 2 == j2 % 2):
            k2 = int(end_y - j2 * K)
        else:
            k2 = int(end_x - i2 * K)
        end_tile = (i2, j2, k2)
        
        # Calculate minimal steps
        steps = minimal_steps(start_tile, end_tile)
        print(steps)

if __name__ == "__main__":
    main()
def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    S = input_data[2:]
    
    # Convert each stand's popcorn flavors to a bitmask
    stands_bits = []
    for i in range(N):
        bitmask = 0
        for j in range(M):
            if S[i][j] == 'o':
                bitmask |= (1 << j)
        stands_bits.append(bitmask)
    
    # All flavors combined bitmask (we want to achieve this)
    all_flavors = (1 << M) - 1
    
    min_stands = N  # at worst, we could visit all stands
    from itertools import combinations
    
    # Since N <= 10, we can check all subsets using bitmasks
    # - or use combinations of stands based on number of stands used
    # We'll do an increasing subset size check, short-circuit as soon as we find a valid coverage
    for size in range(1, N + 1):
        for combo in combinations(range(N), size):
            combined = 0
            for idx in combo:
                combined |= stands_bits[idx]
            if combined == all_flavors:
                min_stands = size
                break
        if min_stands == size:
            break
    
    print(min_stands)

# Do not forget to call main()
main()
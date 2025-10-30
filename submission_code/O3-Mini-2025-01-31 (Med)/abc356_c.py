def main():
    import sys
    input_data = sys.stdin.read().split()
    it = iter(input_data)
    
    # Read N, M, K from input
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    
    # Store test information
    tests = []
    for _ in range(M):
        Ci = int(next(it))
        keys = [int(next(it)) for _ in range(Ci)]
        result = next(it)
        tests.append((keys, result))
    
    count_valid = 0
    # Iterate over all possible combinations of keys (bitmask representation)
    for mask in range(1 << N):
        valid = True
        # For each test, verify the test condition based on current combination (mask)
        for keys, result in tests:
            cnt = 0
            for key in keys:
                # Keys are 1-indexed so adjust for 0-indexed bit positions.
                if mask & (1 << (key - 1)):
                    cnt += 1
            if result == 'o':
                if cnt < K:
                    valid = False
                    break
            elif result == 'x':
                if cnt >= K:
                    valid = False
                    break
        if valid:
            count_valid += 1
    sys.stdout.write(str(count_valid))

if __name__ == '__main__':
    main()
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
        
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    
    # Read tests: each test consists of c keys and result letter.
    tests = []
    for _ in range(M):
        c = int(next(it))
        keys = [int(next(it)) - 1 for _ in range(c)]  # use 0-indexed keys
        result = next(it)
        tests.append((keys, result))
    
    valid_configs = 0
    
    # There are 2^N possible configurations
    for config in range(1 << N):
        valid = True
        for keys, result in tests:
            # Count how many real keys in this test configuration.
            count = 0
            for key in keys:
                if config & (1 << key):
                    count += 1
            # Check test result against the configuration.
            if result == 'o':
                # Door open if at least K real keys inserted.
                if count < K:
                    valid = False
                    break
            else:  # result == 'x'
                # Door closed if fewer than K real keys inserted.
                if count >= K:
                    valid = False
                    break
        if valid:
            valid_configs += 1
    
    print(valid_configs)

if __name__ == "__main__":
    main()
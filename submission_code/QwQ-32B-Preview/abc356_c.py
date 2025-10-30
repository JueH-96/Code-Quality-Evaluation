def main():
    import sys
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    K = int(data[ptr])
    ptr += 1
    tests = []
    for _ in range(M):
        C_i = int(data[ptr])
        ptr += 1
        keys = []
        for _ in range(C_i):
            key = int(data[ptr]) - 1  # Convert to 0-based index
            ptr += 1
            keys.append(key)
        R_i = data[ptr]
        ptr += 1
        # Create bitmask for the keys inserted
        mask = 0
        for key in keys:
            mask |= (1 << key)
        tests.append((mask, R_i))
    # Iterate through all possible combinations of real keys
    count = 0
    for combo in range(1 << N):
        valid = True
        for test in tests:
            mask, R_i = test
            # Number of real keys in the inserted keys
            real_in_test = bin(combo & mask).count('1')
            if R_i == 'o':
                if real_in_test < K:
                    valid = False
                    break
            else:  # R_i == 'x'
                if real_in_test >= K:
                    valid = False
                    break
        if valid:
            count += 1
    print(count)

if __name__ == '__main__':
    main()
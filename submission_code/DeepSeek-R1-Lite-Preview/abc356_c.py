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
            keys.append(int(data[ptr]))
            ptr += 1
        R_i = data[ptr]
        ptr += 1
        # compute test_key_mask
        test_key_mask = 0
        for key in keys:
            test_key_mask |= (1 << (key - 1))
        tests.append((test_key_mask, R_i))
    # now iterate through all possible combination_masks
    count = 0
    for combination_mask in range(0, 1 << N):
        valid = True
        for test_key_mask, R_i in tests:
            count_real = bin(combination_mask & test_key_mask).count('1')
            if R_i == 'o':
                if count_real < K:
                    valid = False
                    break
            else:  # 'x'
                if count_real >= K:
                    valid = False
                    break
        if valid:
            count += 1
    print(count)

if __name__ == '__main__':
    main()
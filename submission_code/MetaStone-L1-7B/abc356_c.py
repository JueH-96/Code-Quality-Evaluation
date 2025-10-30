def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1

    tests = []
    for _ in range(M):
        C_i = int(input[idx])
        idx += 1
        A_i = list(map(int, input[idx:idx + C_i]))
        idx += C_i
        R_i = input[idx]
        idx += 1
        tests.append((A_i, R_i))

    count = 0
    for mask in range(0, 1 << N):
        valid = True
        for keys, res in tests:
            real = 0
            for key in keys:
                if (mask & (1 << (key - 1))):
                    real += 1
            if res == 'o':
                if real < K:
                    valid = False
                    break
            else:
                if real >= K:
                    valid = False
                    break
        if valid:
            count += 1

    print(count)

if __name__ == '__main__':
    main()
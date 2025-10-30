def main():
    import sys
    N, M, K = map(int, sys.stdin.readline().split())
    tests = []
    for _ in range(M):
        parts = sys.stdin.readline().split()
        C_i = int(parts[0])
        A_list = list(map(int, parts[1:-1]))
        R_i = parts[-1].strip()
        mask = 0
        for a in A_list:
            mask |= 1 << (a - 1)
        tests.append((mask, R_i))
    
    count = 0
    for s in range(0, 1 << N):
        valid = True
        for (mask_t, R) in tests:
            common = s & mask_t
            cnt = bin(common).count('1')
            if R == 'o':
                if cnt < K:
                    valid = False
                    break
            else:
                if cnt >= K:
                    valid = False
                    break
        if valid:
            count += 1
    print(count)

if __name__ == "__main__":
    main()
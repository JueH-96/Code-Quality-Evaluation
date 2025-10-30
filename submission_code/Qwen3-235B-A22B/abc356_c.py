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
        A_list = []
        for __ in range(C_i):
            A_list.append(int(input[idx]))
            idx += 1
        R_i = input[idx]
        idx += 1
        mask = 0
        for x in A_list:
            mask |= 1 << (x - 1)
        tests.append((mask, R_i))
    
    ans = 0
    # Iterate all subsets of real keys
    for S in range(1 << N):
        valid = True
        for (mask, R) in tests:
            intersection = S & mask
            cnt = bin(intersection).count('1')
            if R == 'o':
                if cnt >= K:
                    continue
                else:
                    valid = False
                    break
            else:  # R == 'x'
                if cnt < K:
                    continue
                else:
                    valid = False
                    break
        if valid:
            ans += 1
    print(ans)

if __name__ == '__main__':
    main()
def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    
    tests = []
    for _ in range(M):
        C_i = int(input[ptr])
        ptr += 1
        A_i = list(map(int, input[ptr:ptr+C_i]))
        ptr += C_i
        R_i = input[ptr]
        ptr += 1
        
        mask = 0
        for a in A_i:
            mask |= 1 << (a-1)
        flag = (R_i == 'o')
        tests.append((mask, flag))
    
    ans = 0
    for S in range(1 << N):
        valid = True
        for (mask, flag) in tests:
            cnt = bin(S & mask).count('1')
            if (cnt >= K) != flag:
                valid = False
                break
        if valid:
            ans += 1
    print(ans)

if __name__ == "__main__":
    main()
def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1

    test_cases = []
    for _ in range(M):
        C_i = int(input[idx]); idx +=1
        A_i = list(map(int, input[idx:idx+C_i]))
        idx += C_i
        R_i = input[idx]; idx +=1
        test_cases.append( (A_i, R_i) )
    
    count = 0
    for mask in range(1 << N):
        valid = True
        for (A, R) in test_cases:
            cnt = 0
            for a in A:
                if (mask >> (a-1)) & 1:
                    cnt +=1
            if R == 'o':
                if cnt < K:
                    valid = False
                    break
            else:
                if cnt >= K:
                    valid = False
                    break
        if valid:
            count +=1
    print(count)

if __name__ == '__main__':
    main()
def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1

    tests = []
    for _ in range(M):
        C_i = int(input[idx]); idx +=1
        A_list = list(map(int, input[idx:idx+C_i]))
        idx += C_i
        R_i = input[idx]; idx +=1
        # Convert to 0-based keys
        keys = [a-1 for a in A_list]
        required = (R_i == 'o')
        tests.append( (keys, required) )
    
    count = 0
    for mask in range(0, 1 << N):
        valid = True
        for (keys, required) in tests:
            cnt = 0
            for key in keys:
                if (mask >> key) & 1:
                    cnt += 1
            if required:
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
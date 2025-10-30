def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    K = int(data[idx+2])
    idx += 3
    
    tests = []
    for _ in range(M):
        C = int(data[idx])
        A = list(map(int, data[idx+1:idx+1+C]))
        R = data[idx+1+C]
        tests.append((A, R))
        idx += 1 + C + 1
    
    valid_count = 0
    for mask in range(0, 1 << N):
        valid = True
        for test in tests:
            A, R = test
            cnt = 0
            for key in A:
                if mask & (1 << (key-1)):
                    cnt += 1
            if R == 'o':
                if cnt < K:
                    valid = False
                    break
            else:
                if cnt >= K:
                    valid = False
                    break
        if valid:
            valid_count += 1
    
    print(valid_count)

if __name__ == "__main__":
    main()
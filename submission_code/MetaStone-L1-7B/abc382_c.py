def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+M]))
    idx += M
    
    available = [True] * (N + 1)  # 1-based indexing
    result = [-1] * M
    
    for j in range(M):
        Bj = B[j]
        found = False
        for i in range(1, N + 1):
            if available[i] and A[i-1] <= Bj:
                result[j] = i
                available[i] = False
                found = True
                break
        if not found:
            result[j] = -1
    
    for res in result:
        print(res)

if __name__ == '__main__':
    main()
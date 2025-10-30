def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    result = []
    for K in range(N):
        current_sum = A[K]
        # Process left
        for i in range(K-1, -1, -1):
            if A[i] < current_sum:
                current_sum += A[i]
            else:
                break
        # Process right
        for i in range(K+1, N):
            if A[i] < current_sum:
                current_sum += A[i]
            else:
                break
        result.append(current_sum)
    
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    S = sum(A)
    q = S // N
    r = S % N
    
    A.sort()
    
    operations = 0
    
    # First N - r elements should be at least q
    for i in range(N - r):
        if A[i] < q:
            operations += q - A[i]
    
    # Last r elements should be at least q + 1
    for i in range(N - r, N):
        if A[i] < q + 1:
            operations += q + 1 - A[i]
    
    print(operations)

if __name__ == "__main__":
    main()
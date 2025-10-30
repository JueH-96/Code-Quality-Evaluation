def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    for i in range(N-3):
        current = A[i+1] + A[i+2]
        target = A[i] + A[i+3]
        if current > target:
            A[i+1] = (A[i] + A[i+3] - current) // 2
            A[i+2] = (A[i] + A[i+3] + current) // 2
    
    print(sum(A))

if __name__ == '__main__':
    main()
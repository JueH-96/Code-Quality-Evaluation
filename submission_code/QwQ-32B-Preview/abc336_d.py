def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    left = [0] * (N + 1)
    for i in range(2, N + 1):
        m = 1
        while i - m >= 1 and A[i - m] >= m:
            m += 1
        left[i] = m - 1  # last m where condition holds
    
    right = [0] * (N + 1)
    for i in range(N - 1, 0, -1):
        m = 1
        while i + m <= N and A[i + m] >= m:
            m += 1
        right[i] = m - 1  # last m where condition holds
    
    max_k = 0
    for i in range(1, N + 1):
        if i - left[i] >= 1 and i + right[i] <= N:
            k = min(A[i - 1], left[i] + 1, right[i] + 1)
            max_k = max(max_k, k)
    
    print(max_k)

if __name__ == '__main__':
    main()
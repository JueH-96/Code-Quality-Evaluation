def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Compute left_sum
    left_sum = [0] * N
    stack = []
    for i in range(N):
        current = A[i]
        while stack and stack[-1] < current:
            current += stack.pop()
        left_sum[i] = current
        stack.append(current)
    
    # Compute right_sum
    right_sum = [0] * N
    stack = []
    for i in range(N-1, -1, -1):
        current = A[i]
        while stack and stack[-1] < current:
            current += stack.pop()
        right_sum[i] = current
        stack.append(current)
    
    # Calculate the result for each K
    result = [left_sum[i] + right_sum[i] - A[i] for i in range(N)]
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()
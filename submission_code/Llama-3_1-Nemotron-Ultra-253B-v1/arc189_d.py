def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    
    left = [0] * N
    stack = []
    for i in range(N):
        current_sum = A[i]
        current_max = 0
        while stack:
            top = stack[-1]
            if prefix[i] - prefix[top] > 2 * current_max:
                break
            current_sum += prefix[i] - prefix[top+1]
            current_max = max(current_max, prefix[i] - prefix[top])
            stack.pop()
        left[i] = stack[-1] + 1 if stack else 0
        stack.append(i)
    
    right = [N-1] * N
    stack = []
    for i in range(N-1, -1, -1):
        current_sum = A[i]
        current_max = 0
        while stack:
            top = stack[-1]
            if prefix[N] - prefix[i] > 2 * current_max:
                break
            current_sum += prefix[N] - prefix[i+1]
            current_max = max(current_max, prefix[N] - prefix[i+1])
            stack.pop()
        right[i] = stack[-1] - 1 if stack else N-1
        stack.append(i)
    
    result = []
    for i in range(N):
        L = left[i]
        R = right[i]
        total = prefix[R+1] - prefix[L]
        result.append(total)
    
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()
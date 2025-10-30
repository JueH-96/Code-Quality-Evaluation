def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    H = [0] + H  # Convert to 1-based index
    
    # Compute left_greater using a monotonic stack
    left_greater = [0] * (N + 1)
    stack = []
    for j in range(1, N + 1):
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        if stack:
            left_greater[j] = stack[-1]
        else:
            left_greater[j] = 0
        stack.append(j)
    
    # Initialize delta array
    delta = [0] * (N + 2)  # indexes 0..N+1
    
    for j in range(1, N + 1):
        L = left_greater[j]
        if L == 0:
            a = 1
        else:
            a = L
        b = j - 1
        if a <= b:
            delta[a] += 1
            if b + 1 <= N:
                delta[b + 1] -= 1
    
    # Compute prefix sums to get the result
    current = 0
    c = [0] * (N + 1)
    for i in range(1, N + 1):
        current += delta[i]
        c[i] = current
    
    # Prepare the output
    print(' '.join(map(str, c[1:N+1])))

if __name__ == '__main__':
    main()
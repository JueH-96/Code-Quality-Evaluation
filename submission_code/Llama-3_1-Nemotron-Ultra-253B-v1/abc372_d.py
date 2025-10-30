def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    # Compute L[j] using monotonic stack
    stack = []
    L = [-1] * N
    for j in range(N):
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        else:
            L[j] = -1
        stack.append(j)
    
    ans = [0] * N
    delta = [0] * (N + 2)  # To handle end+1 safely
    
    for j in range(N):
        if L[j] < j - 1:
            start = L[j] + 1
            end = j - 1
            if start <= end:
                delta[start] += 1
                delta[end + 1] -= 1
        else:
            if j - 1 >= 0:
                ans[j - 1] += 1
    
    # Apply delta to ans
    current = 0
    for i in range(N):
        current += delta[i]
        ans[i] += current
    
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()
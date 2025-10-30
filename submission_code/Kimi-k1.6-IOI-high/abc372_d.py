def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    # Compute L array using monotonic stack
    stack = []
    L = [-1] * N
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        else:
            L[j] = -1
        stack.append(j)
    
    # Initialize difference array
    diff = [0] * (N + 2)  # 0-based, indices 0..N
    
    for j in range(N):
        start = max(L[j], 0)
        end = j - 1
        if start <= end:
            diff[start] += 1
            diff[end + 1] -= 1
    
    # Compute prefix sums
    res = []
    current = 0
    for i in range(N):
        current += diff[i]
        res.append(current)
    
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()
def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We will find for each building j (0-based),
    # L[j] = the nearest index to the left of j with height > H[j], or -1 if none.
    # Then for j, all i in [L[j], j-1] (clamped to [0..j-1]) are valid. 
    
    # Use a monotonic stack (decreasing by height) to find L[j].
    stack = []
    L = [-1]*N
    for j in range(N):
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        L[j] = stack[-1] if stack else -1
        stack.append(j)
    
    # We'll use a difference array D of length N+1.
    # For each j, we add +1 to D[start] and -1 to D[end+1],
    # where start = max(0, L[j]) and end = j-1, if start <= end.
    D = [0]*(N+1)
    for j in range(N):
        start = max(L[j], 0)
        end = j-1
        if start <= end:
            D[start] += 1
            D[end+1] -= 1
    
    # Now compute the prefix sums of D to get c[i].
    c = [0]*N
    running = 0
    for i in range(N):
        running += D[i]
        c[i] = running
    
    # Print the result (c[i] corresponds to building i+1 in 1-based indexing).
    print(" ".join(map(str, c)))
    
# Let's call solve() to execute.
def __main__():
    solve()
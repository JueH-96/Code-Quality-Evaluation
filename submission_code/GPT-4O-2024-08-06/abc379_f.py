def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    heights = list(map(int, data[2:N+2]))
    queries = []
    index = N + 2
    for _ in range(Q):
        l = int(data[index]) - 1
        r = int(data[index + 1]) - 1
        queries.append((l, r))
        index += 2
    
    # Precompute the next visible building for each building
    next_visible = [-1] * N
    stack = []
    
    for i in range(N-1, -1, -1):
        while stack and heights[stack[-1]] < heights[i]:
            stack.pop()
        if stack:
            next_visible[i] = stack[-1]
        stack.append(i)
    
    # Precompute visible buildings from each building
    visible_from = [[] for _ in range(N)]
    
    for i in range(N):
        j = next_visible[i]
        while j != -1:
            visible_from[i].append(j)
            j = next_visible[j]
    
    # Answer each query
    results = []
    
    for l, r in queries:
        visible_from_r = set(visible_from[r])
        count = 0
        for building in visible_from[l]:
            if building > r and building in visible_from_r:
                count += 1
        results.append(count)
    
    # Print results
    for result in results:
        print(result)
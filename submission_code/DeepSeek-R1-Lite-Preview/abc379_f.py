def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    H = list(map(int, data[2:2+N]))
    queries = []
    for i in range(Q):
        l = int(data[2+N+2*i])
        r = int(data[2+N+2*i+1])
        queries.append((l, r))
    
    # Compute next_taller using a stack
    next_taller = [N+1] * (N+2)
    stack = []
    for i in range(N-1, -1, -1):
        while stack and H[stack[-1]-1] <= H[i]:
            stack.pop()
        if stack:
            next_taller[i+1] = stack[-1]
        stack.append(i+1)
    
    # Build visibility lists for each building
    visibility = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        j = next_taller[i]
        while j <= N:
            visibility[i].append(j)
            j = next_taller[j]
    
    # Process each query
    for l, r in queries:
        vis_list = visibility[l]
        # Use binary search to find the first building > r
        left, right = 0, len(vis_list)
        while left < right:
            mid = (left + right) // 2
            if vis_list[mid] > r:
                right = mid
            else:
                left = mid + 1
        print(len(vis_list) - left)

if __name__ == '__main__':
    main()
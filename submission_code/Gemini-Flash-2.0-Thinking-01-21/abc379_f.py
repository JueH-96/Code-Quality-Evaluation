def solve():
    n, q = map(int, input().split())
    heights = list(map(int, input().split()))
    visibility = [[False] * (n + 1) for _ in range(n + 1)]
    for j in range(2, n + 1):
        current_max_height = -1
        for i in range(j - 1, 0, -1):
            is_visible = (current_max_height <= heights[j-1])
            visibility[i][j] = is_visible
            current_max_height = max(current_max_height, heights[i-1])
            
    queries = []
    for _ in range(q):
        l, r = map(int, input().split())
        queries.append((l, r))
        
    results = []
    for l_query, r_query in queries:
        count = 0
        for j in range(r_query + 1, n + 1):
            if visibility[l_query][j] and visibility[r_query][j]:
                count += 1
        results.append(count)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()
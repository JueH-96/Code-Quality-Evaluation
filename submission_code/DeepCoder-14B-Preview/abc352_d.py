def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    k = int(data[1])
    p = list(map(int, data[2:2+n]))
    
    # Create the pos array where pos[v] is the 1-based index in p where value v is located
    pos = [0] * (n + 1)
    for i in range(n):
        v = p[i]
        pos[v] = i + 1  # positions are 1-based
    
    # Precompute log_table
    log_table = [0] * (n + 1)
    for i in range(2, n + 1):
        log_table[i] = log_table[i // 2] + 1
    
    # Build sparse tables for range min and max queries
    max_level = 0
    while (1 << max_level) <= n:
        max_level += 1
    max_level -= 1
    
    st_min = [[0] * (n + 1) for _ in range(max_level + 1)]
    st_max = [[0] * (n + 1) for _ in range(max_level + 1)]
    
    # Initialize level 0
    for i in range(1, n + 1):
        st_min[0][i] = pos[i]
        st_max[0][i] = pos[i]
    
    # Build higher levels
    for j in range(1, max_level + 1):
        for i in range(1, (n - (1 << j)) + 1):
            st_min[j][i] = min(st_min[j-1][i], st_min[j-1][i + (1 << (j-1))])
            st_max[j][i] = max(st_max[j-1][i], st_max[j-1][i + (1 << (j-1))])
    
    # Function to query min in range [l, r]
    def query_min(l, r):
        if l > r:
            return 0
        length = r - l + 1
        if length == 0:
            return 0
        k = log_table[length]
        return min(st_min[k][l], st_min[k][r - (1 << k) + 1])
    
    # Function to query max in range [l, r]
    def query_max(l, r):
        if l > r:
            return 0
        length = r - l + 1
        if length == 0:
            return 0
        k = log_table[length]
        return max(st_max[k][l], st_max[k][r - (1 << k) + 1])
    
    min_span = float('inf')
    max_a = n - k + 1
    for a in range(1, max_a + 1):
        l = a
        r = a + k - 1
        current_min = query_min(l, r)
        current_max = query_max(l, r)
        span = current_max - current_min
        if span < min_span:
            min_span = span
    
    print(min_span)

if __name__ == "__main__":
    main()
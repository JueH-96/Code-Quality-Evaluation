def solve():
    n, m = map(int, input().split())
    intervals = []
    for _ in range(n):
        l_i, r_i = map(int, input().split())
        intervals.append({'L': l_i, 'R': r_i})
    
    def get_intersection_size(indices):
        if not indices:
            return 0
        min_l = float('inf')
        max_r = float('-inf')
        for index in indices:
            min_l = min(min_l, intervals[index]['L'])
            max_r = max(max_r, intervals[index]['R'])
        
        if max_r > m:
            return 0
            
        count = 0
        start_l = 1
        end_l = min(m, min_l)
        if start_l > end_l:
            return 0
            
        for l in range(start_l, end_l + 1):
            start_r = max(l, max_r)
            end_r = m
            if start_r <= end_r:
                count += (end_r - start_r + 1)
                
        return count
        
    union_size = 0
    for i in range(1, 1 << n):
        current_indices = []
        for j in range(n):
            if (i >> j) & 1:
                current_indices.append(j)
        
        intersection_size = get_intersection_size(current_indices)
        if len(current_indices) % 2 == 1:
            union_size += intersection_size
        else:
            union_size -= intersection_size
            
    total_pairs = m * (m + 1) // 2
    result = total_pairs - union_size
    print(result)

if __name__ == '__main__':
    solve()
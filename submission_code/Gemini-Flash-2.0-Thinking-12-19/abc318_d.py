def solve():
    n = int(input())
    if n < 2:
        print(0)
        return
    weights = [[0] * n for _ in range(n)]
    line_index = 0
    for i in range(n - 1):
        line = list(map(int, input().split()))
        for j in range(i + 1, n):
            weights[i][j] = line[j - (i + 1)]
            weights[j][i] = line[j - (i + 1)]
            
    memo = {}
    
    def get_max_weight_matching(mask):
        if mask == 0:
            return 0
        if mask in memo:
            return memo[mask]
        
        first_vertex_index = -1
        for i in range(n):
            if (mask >> i) & 1:
                first_vertex_index = i
                break
                
        if first_vertex_index == -1:
            return 0
            
        max_weight = get_max_weight_matching(mask & ~(1 << first_vertex_index))
        
        for second_vertex_index in range(first_vertex_index + 1, n):
            if (mask >> second_vertex_index) & 1:
                current_weight = weights[first_vertex_index][second_vertex_index]
                remaining_mask = (mask & ~(1 << first_vertex_index)) & ~(1 << second_vertex_index)
                current_weight += get_max_weight_matching(remaining_mask)
                max_weight = max(max_weight, current_weight)
                
        memo[mask] = max_weight
        return max_weight
        
    initial_mask = (1 << n) - 1
    result = get_max_weight_matching(initial_mask)
    print(result)

if __name__ == '__main__':
    solve()
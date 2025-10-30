import sys

def solve():
    n = int(sys.stdin.readline())
    if n < 2:
        print(0)
        return
    weights = {}
    for i in range(1, n):
        line_weights = list(map(int, sys.stdin.readline().split()))
        for j in range(i + 1, n + 1):
            weights[(i, j)] = line_weights[j - i - 1]
            
    memo = {}
    
    def get_max_weight_matching(mask):
        if mask == 0:
            return 0
        if mask in memo:
            return memo[mask]
        
        vertices_in_mask = []
        for i in range(n):
            if (mask >> i) & 1:
                vertices_in_mask.append(i + 1)
        
        if len(vertices_in_mask) <= 1:
            return 0
            
        first_vertex = vertices_in_mask[0]
        
        # Option 1: Don't use first_vertex in any edge
        max_weight = get_max_weight_matching(mask & ~(1 << (first_vertex - 1)))
        
        # Option 2: Pair first_vertex with another vertex
        for i in range(1, len(vertices_in_mask)):
            second_vertex = vertices_in_mask[i]
            edge_weight = weights.get((min(first_vertex, second_vertex), max(first_vertex, second_vertex)), 0)
            remaining_mask = mask & ~(1 << (first_vertex - 1)) & ~(1 << (second_vertex - 1))
            current_weight = edge_weight + get_max_weight_matching(remaining_mask)
            max_weight = max(max_weight, current_weight)
            
        memo[mask] = max_weight
        return max_weight
        
    initial_mask = (1 << n) - 1
    result = get_max_weight_matching(initial_mask)
    print(result)

if __name__ == '__main__':
    solve()
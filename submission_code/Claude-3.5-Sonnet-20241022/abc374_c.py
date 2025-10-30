def solve_min_max_group(N, K):
    total_people = sum(K)
    min_max_group = float('inf')
    
    def try_combinations(index, group_a_sum):
        nonlocal min_max_group
        
        # Base case: all departments have been assigned
        if index == N:
            group_b_sum = total_people - group_a_sum
            max_group = max(group_a_sum, group_b_sum)
            min_max_group = min(min_max_group, max_group)
            return
            
        # Try adding current department to group A
        try_combinations(index + 1, group_a_sum + K[index])
        
        # Try adding current department to group B
        try_combinations(index + 1, group_a_sum)
    
    try_combinations(0, 0)
    return min_max_group

# Read input
N = int(input())
K = list(map(int, input().split()))

# Solve and print result
result = solve_min_max_group(N, K)
print(result)
n = int(input())
k = list(map(int, input().split()))

total_sum = sum(k)
min_max = float('inf')

# Try all possible subsets for Group A (using bit manipulation)
for mask in range(1 << n):
    group_a_sum = 0
    
    # Calculate sum for Group A based on the mask
    for i in range(n):
        if mask & (1 << i):
            group_a_sum += k[i]
    
    # Group B sum is the remaining
    group_b_sum = total_sum - group_a_sum
    
    # Find the maximum of the two groups
    current_max = max(group_a_sum, group_b_sum)
    
    # Update minimum of the maximums
    min_max = min(min_max, current_max)

print(min_max)
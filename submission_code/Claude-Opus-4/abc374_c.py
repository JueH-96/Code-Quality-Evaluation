# YOUR CODE HERE
n = int(input())
k = list(map(int, input().split()))

total_sum = sum(k)
min_max = float('inf')

# Try all possible assignments using bitmask
for mask in range(1 << n):
    group_a_sum = 0
    
    # Calculate sum for Group A (departments with bit 1)
    for i in range(n):
        if mask & (1 << i):
            group_a_sum += k[i]
    
    # Group B sum is total - Group A sum
    group_b_sum = total_sum - group_a_sum
    
    # Maximum of the two groups
    max_sum = max(group_a_sum, group_b_sum)
    
    # Update minimum
    min_max = min(min_max, max_sum)

print(min_max)
# YOUR CODE HERE
n = int(input())
a = [0] + list(map(int, input().split()))  # 1-indexed for easier handling

# Precompute prefix distinct counts
# prefix_distinct[i] = number of distinct integers in a[1:i]
prefix_distinct = [0] * (n + 1)
seen = set()
for i in range(1, n + 1):
    seen.add(a[i])
    prefix_distinct[i] = len(seen)

# Precompute suffix distinct counts
# suffix_distinct[i] = number of distinct integers in a[i:n]
suffix_distinct = [0] * (n + 2)
seen = set()
for i in range(n, 0, -1):
    seen.add(a[i])
    suffix_distinct[i] = len(seen)

max_sum = 0

# Try all possible values of j (end of middle subarray)
for j in range(2, n):  # j can be from 2 to n-1
    middle_set = set()
    # Try all possible values of i (start of middle subarray)
    for i in range(j - 1, 0, -1):  # i can be from 1 to j-1
        # Add the leftmost element of current middle subarray
        middle_set.add(a[i + 1])
        
        # Calculate sum of distinct counts in three subarrays
        current_sum = prefix_distinct[i] + len(middle_set) + suffix_distinct[j + 1]
        max_sum = max(max_sum, current_sum)

print(max_sum)
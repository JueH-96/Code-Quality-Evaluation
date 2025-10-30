# YOUR CODE HERE
S = input().strip()
n = len(S)

# For each character, maintain count and sum of indices
char_count = {}
char_sum = {}

total = 0
for k in range(n):
    c = S[k]
    
    # Calculate contribution for this k
    count = char_count.get(c, 0)
    sum_indices = char_sum.get(c, 0)
    total += k * count - sum_indices - count
    
    # Update count and sum
    char_count[c] = count + 1
    char_sum[c] = sum_indices + k

print(total)
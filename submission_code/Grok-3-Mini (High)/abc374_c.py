import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = [int(x) for x in data[index:index + N]]
T = sum(K)
min_max_val = T  # Initialize to maximum possible
for mask in range(1 << N):  # Iterate over all possible subsets
    sum_A = 0
    for i in range(N):
        if mask & (1 << i):  # Check if i-th bit is set
            sum_A += K[i]
    sum_B = T - sum_A  # Sum of the other group
    current_max = max(sum_A, sum_B)  # Maximum of the two sums
    if current_max < min_max_val:
        min_max_val = current_max  # Update minimum of maximums
print(min_max_val)  # Output the result
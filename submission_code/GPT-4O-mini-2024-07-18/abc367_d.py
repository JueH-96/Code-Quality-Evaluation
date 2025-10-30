def count_pairs(N, M, A):
    # Calculate prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]

    # Count valid pairs
    count = 0
    mod_count = {}
    
    for s in range(1, N + 1):
        # We want to find t such that (prefix_sum[t] - prefix_sum[s]) % M == 0
        # This means prefix_sum[t] % M == prefix_sum[s] % M
        current_mod = prefix_sum[s] % M
        
        # Count how many times this mod has been seen
        if current_mod in mod_count:
            count += mod_count[current_mod]
        
        # Update the mod_count for the current prefix_sum
        if current_mod in mod_count:
            mod_count[current_mod] += 1
        else:
            mod_count[current_mod] = 1
    
    return count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

result = count_pairs(N, M, A)
print(result)
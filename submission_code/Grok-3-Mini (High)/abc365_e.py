import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N

# Compute prefix XOR array
prefix = [0]
for num in A:
    prefix.append(prefix[-1] ^ num)

# Initialize sum answer
sum_ans = 0

# Iterate over each bit (0 to 30)
for bit in range(31):
    # Extract the bit values for the prefix XOR array
    B = [(p >> bit) & 1 for p in prefix]
    
    # Compute suffix sum of B (sum of B[m] from m=k to N)
    suff_B_sum = [0] * (N + 1)
    suff_B_sum[N] = B[N]
    for k in range(N - 1, -1, -1):
        suff_B_sum[k] = B[k] + suff_B_sum[k + 1]
    
    # Compute the count of subarrays where the bit is set in XOR
    cnt_b = 0
    for I in range(0, N - 1):  # I from 0 to N-2 inclusive
        if B[I] == 0:
            cnt_b += suff_B_sum[I + 2]
        else:
            cnt_b += (N - I - 1) - suff_B_sum[I + 2]
    
    # Add the contribution of this bit to the sum
    sum_ans += cnt_b * (1 << bit)

# Output the result
print(sum_ans)
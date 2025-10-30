n = int(input())
a = list(map(int, input().split()))

# Calculate prefix XOR
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i + 1] = prefix[i] ^ a[i]

total = 0
# For each subarray from i to j (1-indexed)
for i in range(1, n + 1):
    for j in range(i + 1, n + 1):
        # XOR of subarray from i to j is prefix[j] ^ prefix[i-1]
        subarray_xor = prefix[j] ^ prefix[i - 1]
        total += subarray_xor

print(total)
# Read input
n, k = map(int, input().split())
a = list(map(int, input().split()))

# Create a set to remove duplicates
s = set(a)

# Calculate sum of unique elements in A that are <= K
sum_in = 0
for x in s:
    if x <= k:
        sum_in += x

# Calculate total sum from 1 to K
total = k * (k + 1) // 2

# The result is total sum minus sum of elements present in A and <= K
print(total - sum_in)
n, k = map(int, input().split())
s = input().strip()

# Find all 1-blocks
blocks = []
i = 0
while i < n:
    if s[i] == '1':
        start = i
        while i < n and s[i] == '1':
            i += 1
        blocks.append((start + 1, i))  # Store as 1-indexed [start, end]
    else:
        i += 1

# Get the relevant blocks
r_k_minus_1 = blocks[k - 2][1]
l_k = blocks[k - 1][0]
r_k = blocks[k - 1][1]

# Build the result
result = []
for i in range(1, n + 1):
    if i <= r_k_minus_1:
        result.append(s[i - 1])
    elif i <= r_k_minus_1 + (r_k - l_k + 1):
        result.append('1')
    elif i <= r_k:
        result.append('0')
    else:
        result.append(s[i - 1])

print(''.join(result))
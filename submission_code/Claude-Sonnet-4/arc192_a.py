n = int(input())
a = list(map(int, input().split()))

# Find all maximal blocks of consecutive 0s
blocks = []
i = 0
while i < n:
    if a[i] == 0:
        start = i
        while i < n and a[i] == 0:
            i += 1
        blocks.append(i - start)
    else:
        i += 1

# Check if all blocks can be covered
possible = True
for block_len in blocks:
    if block_len == 1 or (block_len >= 3 and block_len % 2 == 1):
        possible = False
        break

print("Yes" if possible else "No")
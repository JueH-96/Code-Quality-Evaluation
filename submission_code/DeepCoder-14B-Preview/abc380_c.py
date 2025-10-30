# Read input
N, K = map(int, input().split())
S = input().strip()

# Extract all 1-blocks
blocks = []
i = 0
while i < len(S):
    if S[i] == '1':
        if i == 0 or S[i-1] == '0':
            start = i
            j = i
            while j < len(S) and S[j] == '1':
                j += 1
            end = j - 1
            if end == len(S) - 1 or S[end + 1] == '0':
                blocks.append((start, end))
        i = j
    else:
        i += 1

# Get the required blocks
block_prev = blocks[K-2]
block_curr = blocks[K-1]

# Extract parts of the string
part1 = S[:block_prev[1]+1]
part2 = S[block_curr[0]:block_curr[1]+1]
part3 = S[block_prev[1]+1: block_curr[0]]
part4 = S[block_curr[1]+1:]

# Construct the new string
new_S = part1 + part2 + part3 + part4

print(new_S)
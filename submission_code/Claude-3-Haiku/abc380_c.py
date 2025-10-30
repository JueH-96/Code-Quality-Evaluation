# Read the input
N, K = map(int, input().split())
S = input()

# Find the 1-blocks in the string
blocks = []
start = 0
while start < N:
    end = start
    while end < N and S[end] == '1':
        end += 1
    if end > start:
        blocks.append((start, end-1))
    start = end + 1

# Move the K-th 1-block to after the (K-1)-th 1-block
result = ''
for i in range(len(blocks)):
    if i == K-1:
        result += S[blocks[i][0]:blocks[i-1][1]+1]
        result += S[blocks[i][0]:blocks[i][1]+1]
    elif i < K-1:
        result += S[blocks[i][0]:blocks[i][1]+1]
    else:
        result += S[blocks[i][0]:blocks[i][1]+1]

# Print the result
print(result)
import math

# Read N
N = int(input())

# Calculate minimum number of friends needed (ceil(log2(N)))
M = math.ceil(math.log2(N))
print(M)

# For each friend i, give them bottles where bit i in the binary representation is 1
for i in range(M):
    bottles = []
    for j in range(1, N+1):
        if (j-1) & (1 << i):
            bottles.append(j)
    print(len(bottles), *bottles)

# Read results
S = input()

# Convert binary string to decimal to get spoiled bottle number
# Add 1 since bottles are 1-indexed
ans = 1
for i in range(M):
    if S[i] == '1':
        ans += (1 << i)
print(ans)
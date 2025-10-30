n = int(input())

# Determine the minimal number of friends M such that 2^M > n
m = 0
while (1 << m) <= n:
    m += 1

# Generate the list of friends' bottles
friends = [[] for _ in range(m)]
for x in range(1, n + 1):
    binary = bin(x)[2:].zfill(m)
    for i in range(m):
        if binary[i] == '1':
            friends[i].append(x)

# Read the response string S
s = input().strip()

# Compute X from the response string
x = 0
for c in s:
    x = (x << 1) | (1 if c == '1' else 0)

print(x)
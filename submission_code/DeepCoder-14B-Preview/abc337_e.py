n = int(input())

# Calculate the minimal number of friends M
M = 1
while (2 ** M) < n:
    M += 1

# Prepare the distribution for each friend
distribution = []
for i in range(1, M + 1):
    bottles = []
    for x in range(1, n + 1):
        code = x - 1
        if (code & (1 << (i - 1))) != 0:
            bottles.append(x)
    bottles.sort()
    distribution.append(bottles)

# Output the number of friends
print(M)

# Output each friend's distribution
for i in range(M):
    bottles = distribution[i]
    k = len(bottles)
    if k == 0:
        print(0)
    else:
        print(f"{k} {' '.join(map(str, bottles))}")

# Read the response string
s = input().strip()

# Calculate the sum based on the response
total = 0
for i in range(len(s)):
    if s[i] == '1':
        total += (1 << i)

# Determine the spoiled bottle
x = total + 1
print(x)
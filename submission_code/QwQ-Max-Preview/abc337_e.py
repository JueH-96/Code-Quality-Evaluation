n = int(input())
m = 0
while (1 << m) < n:
    m += 1

# Generate binary codes for each bottle (0-based index)
codes = []
for i in range(n):
    binary = bin(i)[2:].zfill(m)
    codes.append(binary)

# Prepare friends' lists
friends = []
for j in range(m):
    lst = []
    for i in range(n):
        if codes[i][j] == '1':
            lst.append(i + 1)  # bottle numbers are 1-based
    friends.append(lst)

# Output the number of friends and their assignments
print(m)
for lst in friends:
    print(f"{len(lst)} {' '.join(map(str, lst))}")

# Read the response and determine X
s = input().strip()
y = int(s, 2)
x = y + 1
print(x)
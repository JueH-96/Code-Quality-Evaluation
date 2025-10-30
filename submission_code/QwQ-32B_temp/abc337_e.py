import sys

n = int(sys.stdin.readline())
m = (n - 1).bit_length()

for j in range(m):
    bottles = []
    for x in range(1, n + 1):
        if ( (x - 1) & (1 << j) ):
            bottles.append(x)
    bottles.sort()
    print(len(bottles), end=' ')
    print(' '.join(map(str, bottles)))

s = sys.stdin.readline().strip()
x = 0
for i in range(m):
    bit = int(s[i])
    x += bit * (1 << i)
print(x + 1)
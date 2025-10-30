import re

# Read the input
s = input()
n = int(input())

# Create a list of all possible values in T
t = []
for i in range(2 ** s.count('?')):
    binary = format(i, '0' + str(s.count('?')) + 'b')
    value = ''
    for c in s:
        if c == '?':
            value += binary[0]
            binary = binary[1:]
        else:
            value += c
    t.append(int(value, 2))

# Find the greatest value in T less than or equal to N
t.sort()
for x in reversed(t):
    if x <= n:
        print(x)
        break
else:
    print(-1)
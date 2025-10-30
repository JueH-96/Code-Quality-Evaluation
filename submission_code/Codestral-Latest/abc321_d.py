import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
P = int(data[2])

A = list(map(int, data[3:3+N]))
B = list(map(int, data[3+N:]))

total_price = 0

for a in A:
    for b in B:
        s = a + b
        total_price += min(s, P)

print(total_price)
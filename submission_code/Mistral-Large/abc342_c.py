import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = list(data[1])
Q = int(data[2])

operations = data[3:]

for i in range(Q):
    c, d = operations[2*i], operations[2*i + 1]
    S = [d if char == c else char for char in S]

print(''.join(S))
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:]))

max_score = max(P[1:])
x = max(0, max_score - P[0] + 1)

print(x)
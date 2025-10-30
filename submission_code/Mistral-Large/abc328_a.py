import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
S = list(map(int, data[2:]))

total_score = sum(s for s in S if s <= X)

print(total_score)
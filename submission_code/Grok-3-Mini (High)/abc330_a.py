import sys
data = sys.stdin.read().split()
N = int(data[0])
L = int(data[1])
scores = list(map(int, data[2:2+N]))
count = sum(score >= L for score in scores)
print(count)
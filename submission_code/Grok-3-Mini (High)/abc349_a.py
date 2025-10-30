import sys
data = sys.stdin.read().split()
N = int(data[0])
A_sum = sum(map(int, data[1:]))
print(-A_sum)
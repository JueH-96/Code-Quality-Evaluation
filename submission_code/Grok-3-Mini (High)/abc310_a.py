import sys
data = sys.stdin.read().split()
N = int(data[0])
P = int(data[1])
Q = int(data[2])
D_list = list(map(int, data[3:3+N]))
min_D = min(D_list)
min_cost = min(P, Q + min_D)
print(min_cost)
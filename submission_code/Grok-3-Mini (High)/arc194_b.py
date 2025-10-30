import sys

class Fenwick:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def add(self, index, val):
        while index <= self.size:
            self.tree[index] += val
            index += index & -index
    
    def prefix_sum(self, index):
        sum_val = 0
        i = index
        while i > 0:
            sum_val += self.tree[i]
            i -= i & -i
        return sum_val

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
P = [int(data[i]) for i in range(index, index + N)]

ft = Fenwick(N)
ft.add(P[0], 1)
sum_cost = 0

for k in range(2, N + 1):
    value = P[k - 1]
    if value > 1:
        less = ft.prefix_sum(value - 1)
    else:
        less = 0
    p = less + 1
    D = k - p
    if D > 0:
        sum_cost += (p + (k - 1)) * D // 2
    ft.add(value, 1)

print(sum_cost)
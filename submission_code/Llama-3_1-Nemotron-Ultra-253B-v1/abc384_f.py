import sys
from collections import defaultdict

class FenwickTree:
    def __init__(self, size):
        self.size = size + 2
        self.tree = [0] * (self.size + 1)
    
    def update(self, idx, value):
        idx += 1
        while idx <= self.size:
            self.tree[idx] += value
            idx += idx & -idx
    
    def query(self, idx):
        idx += 1
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res
    
    def query_range(self, l, r):
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)

def compute_sum(A):
    if not A:
        return 0
    groups = defaultdict(list)
    for x in A:
        c = 0
        while x % 2 == 0:
            x //= 2
            c += 1
        groups[c].append(x)
    
    max_c = max(groups.keys()) if groups else 0
    ft_sum = FenwickTree(max_c)
    ft_cnt = FenwickTree(max_c)
    ft_sum_mul = FenwickTree(max_c)
    ft_cnt_mul = FenwickTree(max_c)
    
    sum1 = 0
    for x in A:
        c = 0
        o = x
        while o % 2 == 0:
            o //= 2
            c += 1
        
        sum_less = ft_sum.query(c - 1)
        sum_mul = ft_sum_mul.query(c - 1)
        sum_greater = ft_sum.query_range(c + 1, max_c)
        cnt_greater = ft_cnt.query_range(c + 1, max_c)
        
        contribution = sum_less + (o * (1 << c)) * sum_mul
        contribution += (sum_greater >> c) + o * cnt_greater
        sum1 += contribution
        
        ft_sum.update(c, o)
        ft_sum_mul.update(c, o)
        ft_cnt.update(c, 1)
        ft_sum_mul.update(c, o * (1 << c))
    
    sum2 = 0
    for c in groups:
        new_A = []
        for o in groups[c]:
            new_A.append(o)
        sum2 += compute_sum(new_A)
    
    return sum1 + sum2

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    print(compute_sum(A))

if __name__ == "__main__":
    main()
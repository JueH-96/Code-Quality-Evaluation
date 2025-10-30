import sys

class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, index, delta):
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i
            
    def prefix_query(self, index):
        if index < 0:
            return 0
        i = index + 1
        s = 0
        while i:
            s += self.tree[i]
            i -= i & -i
        return s
    
    def range_query(self, l, r):
        if l > r:
            return 0
        return self.prefix_query(r) - self.prefix_query(l - 1)

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    fenw_val = Fenw(m)
    init_inv = 0
    for a in A:
        init_inv += fenw_val.range_query(a + 1, m - 1)
        fenw_val.update(a, 1)
        
    print(init_inv)
    
    L = [[] for _ in range(m)]
    for i in range(n):
        a = A[i]
        L[a].append(i)
        
    ft_pos_all = Fenw(n)
    ft_index_type2 = Fenw(n)
    for i in range(n):
        ft_pos_all.update(i, 1)
        
    inv1 = init_inv
    inv2 = 0
    cross = 0
    
    for k in range(0, m - 1):
        x = m - k - 1
        lst = L[x]
        for idx_in_list, p in enumerate(lst):
            count_after = len(lst) - idx_in_list - 1
            right_count = ft_pos_all.range_query(p + 1, n - 1)
            right_less = right_count - count_after
            inv1 -= right_less
            
            cross_loss = ft_index_type2.range_query(p + 1, n - 1)
            cross -= cross_loss
            
            ft_pos_all.update(p, -1)
            
            new_inv2 = ft_index_type2.prefix_query(p - 1)
            inv2 += new_inv2
            
            gain = ft_pos_all.prefix_query(p - 1)
            cross += gain
            
            ft_index_type2.update(p, 1)
            
        total = inv1 + inv2 + cross
        print(total)

if __name__ == "__main__":
    main()
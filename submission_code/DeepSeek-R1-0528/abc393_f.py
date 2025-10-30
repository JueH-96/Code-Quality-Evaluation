import sys
from collections import defaultdict

class Fenw:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (self.n + 1)
    
    def update(self, i, val):
        while i <= self.n:
            if val > self.tree[i]:
                self.tree[i] = val
            i += i & -i
            
    def query(self, i):
        res = 0
        while i:
            if self.tree[i] > res:
                res = self.tree[i]
            i -= i & -i
        return res

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    queries = []
    all_vals = set()
    for a in A:
        all_vals.add(a)
    for _ in range(q):
        r = int(next(it))
        x = int(next(it))
        queries.append((r, x))
        all_vals.add(x)
    
    sorted_vals = sorted(all_vals)
    comp_map = {}
    for idx, val in enumerate(sorted_vals, start=1):
        comp_map[val] = idx
    M = len(sorted_vals)
    
    queries_by_r = defaultdict(list)
    for idx, (r, x) in enumerate(queries):
        queries_by_r[r].append((x, idx))
    
    fenw_tree = Fenw(M)
    ans_arr = [0] * q
    
    for i in range(n):
        a_val = A[i]
        comp_a = comp_map[a_val]
        pre_max = fenw_tree.query(comp_a - 1) if comp_a > 1 else 0
        new_val = pre_max + 1
        fenw_tree.update(comp_a, new_val)
        
        r_index = i + 1
        if r_index in queries_by_r:
            for x, orig_idx in queries_by_r[r_index]:
                comp_x = comp_map[x]
                res = fenw_tree.query(comp_x)
                ans_arr[orig_idx] = res
                
    for ans in ans_arr:
        print(ans)

if __name__ == '__main__':
    main()
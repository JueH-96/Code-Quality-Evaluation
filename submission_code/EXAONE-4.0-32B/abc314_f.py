import sys
from collections import deque

mod = 998244353

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    matches = []
    index = 1
    for i in range(n-1):
        p = int(data[index])
        q = int(data[index+1])
        index += 2
        matches.append((p, q))
    
    total_nodes = 2 * n
    parent_dsu = list(range(total_nodes + 1))
    size_dsu = [0] * (total_nodes + 1)
    children = [[] for _ in range(total_nodes + 1)]
    
    for i in range(1, n + 1):
        size_dsu[i] = 1
        
    def find(x):
        if parent_dsu[x] != x:
            parent_dsu[x] = find(parent_dsu[x])
        return parent_dsu[x]
    
    next_id = n + 1
    for p, q in matches:
        r1 = find(p)
        r2 = find(q)
        w = next_id
        next_id += 1
        children[w] = [r1, r2]
        parent_dsu[r1] = w
        parent_dsu[r2] = w
        size_dsu[w] = size_dsu[r1] + size_dsu[r2]
        parent_dsu[w] = w
        
    root = next_id - 1
    res = [0] * (n + 1)
    qq = deque()
    qq.append((root, 0))
    
    while qq:
        node, acc = qq.popleft()
        if node <= n:
            res[node] = acc % mod
        else:
            l, r = children[node]
            a = size_dsu[l]
            b = size_dsu[r]
            total_size = a + b
            inv_total = pow(total_size, mod - 2, mod)
            frac_left = a * inv_total % mod
            frac_right = b * inv_total % mod
            qq.append((l, (acc + frac_left) % mod))
            qq.append((r, (acc + frac_right) % mod))
            
    out_str = " ".join(str(res[i]) for i in range(1, n + 1))
    print(out_str)

if __name__ == "__main__":
    main()
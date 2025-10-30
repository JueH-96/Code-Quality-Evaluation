import sys
sys.setrecursionlimit(300000)

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    p = list(map(int, data[1:1+n]))
    
    pos = [0] * (n + 1)
    for i in range(n):
        pos[p[i]] = i + 1
        
    size = 1
    while size < n:
        size *= 2
    seg = [0] * (2 * size)
    lazy = [0] * (2 * size)
    
    for i in range(1, n + 1):
        seg[size + i - 1] = i
    for i in range(size - 1, 0, -1):
        seg[i] = seg[2 * i] + seg[2 * i + 1]
        
    def push_down(node, l, r):
        if lazy[node] != 0:
            seg[node] += lazy[node] * (r - l + 1)
            if l != r:
                lazy[2 * node] += lazy[node]
                lazy[2 * node + 1] += lazy[node]
            lazy[node] = 0
            
    def update_range(l, r, val, node=1, l_bound=1, r_bound=size):
        push_down(node, l_bound, r_bound)
        if r < l_bound or r_bound < l:
            return
        if l <= l_bound and r_bound <= r:
            lazy[node] += val
            push_down(node, l_bound, r_bound)
            return
        mid = (l_bound + r_bound) // 2
        update_range(l, r, val, 2 * node, l_bound, mid)
        update_range(l, r, val, 2 * node + 1, mid + 1, r_bound)
        seg[node] = seg[2 * node] + seg[2 * node + 1]
        
    def update_point(pos, val):
        node = size + pos - 1
        seg[node] = val
        node //= 2
        while node:
            seg[node] = seg[2 * node] + seg[2 * node + 1]
            node //= 2
            
    def query(l, r, node=1, l_bound=1, r_bound=size):
        push_down(node, l_bound, r_bound)
        if r < l_bound or r_bound < l:
            return 0
        if l <= l_bound and r_bound <= r:
            return seg[node]
        mid = (l_bound + r_bound) // 2
        left_res = query(l, r, 2 * node, l_bound, mid)
        right_res = query(l, r, 2 * node + 1, mid + 1, r_bound)
        return left_res + right_res
        
    total_cost = 0
    for x in range(1, n + 1):
        p_val = pos[x]
        s = query(1, p_val - 1)
        total_cost += s
        update_point(p_val, 0)
        if p_val < n:
            update_range(p_val + 1, n, -1)
            
    print(total_cost)

if __name__ == '__main__':
    main()
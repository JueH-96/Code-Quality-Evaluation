import sys
sys.setrecursionlimit(1 << 25)

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    contests = []
    for _ in range(n):
        L = int(next(it))
        R = int(next(it))
        contests.append((L, R))
    q = int(next(it))
    queries = [int(next(it)) for _ in range(q)]
    
    max_val = 500000
    s = list(range(max_val + 1))
    
    size = 1
    while size < max_val + 1:
        size *= 2
    tree_min = [10**9] * (2 * size)
    tree_max = [-10**9] * (2 * size)
    
    for i in range(max_val + 1):
        tree_min[size + i] = s[i]
        tree_max[size + i] = s[i]
    for i in range(max_val + 1, size):
        tree_min[size + i] = 10**9
        tree_max[size + i] = -10**9
        
    for i in range(size - 1, 0, -1):
        tree_min[i] = min(tree_min[2 * i], tree_min[2 * i + 1])
        tree_max[i] = max(tree_max[2 * i], tree_max[2 * i + 1])
        
    for (L, R) in contests:
        stack = [1]
        updated_leaves = []
        while stack:
            node = stack.pop()
            if tree_min[node] > R or tree_max[node] < L:
                continue
            if node >= size:
                x = node - size
                if x <= max_val and L <= s[x] <= R:
                    updated_leaves.append(x)
                continue
            stack.append(2 * node + 1)
            stack.append(2 * node)
        
        for x in updated_leaves:
            s[x] += 1
            
        for x in updated_leaves:
            idx = size + x
            tree_min[idx] = s[x]
            tree_max[idx] = s[x]
            idx //= 2
            while idx:
                tree_min[idx] = min(tree_min[2 * idx], tree_min[2 * idx + 1])
                tree_max[idx] = max(tree_max[2 * idx], tree_max[2 * idx + 1])
                idx //= 2
                
    out_lines = []
    for x in queries:
        out_lines.append(str(s[x]))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()
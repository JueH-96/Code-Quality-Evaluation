def solve():
    t = int(input())
    for _ in range(t):
        n, x, k = map(int, input().split())

        def get_ancestors(node):
            ancestors = []
            while node > 0:
                ancestors.append(node)
                node //= 2
            return ancestors

        x_ancestors = get_ancestors(x)
        
        count = 0
        
        q = [(x, 0)]
        visited = {x}
        
        while q:
            curr, dist = q.pop(0)
            
            if dist == k:
                count += 1
                continue
            
            if dist > k:
                continue
            
            children = []
            if 2 * curr <= n:
                children.append(2 * curr)
            if 2 * curr + 1 <= n:
                children.append(2 * curr + 1)
            
            parent = curr // 2 if curr > 1 else 0
            if parent > 0:
                children.append(parent)
            
            for child in children:
                if child not in visited:
                    visited.add(child)
                    q.append((child, dist + 1))
        
        print(count)

solve()
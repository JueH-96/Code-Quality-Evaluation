# YOUR CODE HERE
def solve():
    n, x, k = map(int, input().split())

    def get_parent(u):
        return u // 2

    def get_children(u, n):
        children = []
        c1 = u * 2
        if c1 <= n:
            children.append(c1)
        c2 = u * 2 + 1
        if c2 <= n:
            children.append(c2)
        return children

    def dist(u, v, n):
        if u == v:
            return 0
        
        u_parents = {u}
        curr = u
        while curr > 1:
            curr = get_parent(curr)
            u_parents.add(curr)
        
        v_parents = {v}
        curr_dist = 0
        curr = v
        while curr > 1:
            curr = get_parent(curr)
            v_parents.add(curr)
            curr_dist += 1
            if curr in u_parents:
                u_dist = 0
                curr_u = u
                while curr_u != curr:
                    curr_u = get_parent(curr_u)
                    u_dist += 1
                return u_dist + curr_dist

        u_dist = 0
        curr_u = u
        while curr_u > 1:
            curr_u = get_parent(curr_u)
            u_dist += 1
        
        v_dist = 0
        curr_v = v
        while curr_v > 1:
            curr_v = get_parent(curr_v)
            v_dist += 1
            
        return u_dist + v_dist

    count = 0
    for i in range(1, int(n) + 1):
        if dist(x, i, n) == k:
            count += 1
    print(count)


t = int(input())
for _ in range(t):
    solve()
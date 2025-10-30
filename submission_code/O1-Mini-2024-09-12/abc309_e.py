# YOUR CODE HERE
import sys, sys
import sys
import sys
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import defaultdict
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    adj = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        p = int(data[idx]); idx +=1
        adj[p].append(i)
    in_order = [0]*(N+1)
    out_order = [0]*(N+1)
    depth = [0]*(N+1)
    nodes_at_depth = [[] for _ in range(N+1)]
    stack = [('enter', 1, 0)]
    timer = 1
    max_depth = 0
    while stack:
        step = stack.pop()
        if step[0] == 'enter':
            node, d = step[1], step[2]
            in_order[node] = timer
            if d >= len(nodes_at_depth):
                nodes_at_depth.append([])
            nodes_at_depth[d].append(node)
            depth[node] = d
            if d > max_depth:
                max_depth = d
            timer +=1
            stack.append( ('exit', node, d) )
            for child in reversed(adj[node]):
                stack.append( ('enter', child, d+1) )
        else:
            node, d = step[1], step[2]
            out_order[node] = timer -1
    # Read M insurances
    insurances = []
    y_max =0
    for _ in range(M):
        x = int(data[idx]); idx +=1
        y = int(data[idx]); idx +=1
        insurances.append( (x, y) )
        if y > y_max:
            y_max = y
    total_depth = max_depth + y_max +2
    events = [[] for _ in range(total_depth)]
    for x, y in insurances:
        d_start = depth[x]
        d_end = depth[x] + y
        if d_start < len(events):
            events[d_start].append( (in_order[x], out_order[x], 1) )
        if d_end +1 < len(events):
            events[d_end +1].append( (in_order[x], out_order[x], -1) )
    # BIT with range add and point query
    class BIT:
        def __init__(self, size):
            self.N = size
            self.tree = [0]*(self.N +2)
        def update(self, idx, val):
            while idx <= self.N:
                self.tree[idx] += val
                idx += idx & -idx
        def range_add(self, l, r, val):
            self.update(l, val)
            self.update(r+1, -val)
        def point_query(self, idx):
            res =0
            while idx >0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    bit = BIT(N+2)
    covered = [False]*(N+1)
    for d in range(total_depth):
        for event in events[d]:
            l, r, typ = event
            bit.range_add(l, r, typ)
        if d <= max_depth:
            for node in nodes_at_depth[d]:
                if not covered[node] and bit.point_query(in_order[node]) >0:
                    covered[node] = True
    count = sum(covered)
    print(count)

if __name__ == "__main__":
    main()
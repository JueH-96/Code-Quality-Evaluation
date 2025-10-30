import sys
sys.setrecursionlimit(300000)
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    graph = [[] for _ in range(n+1)]
    index = 1
    for i in range(n-1):
        u = int(data[index]); v = int(data[index+1]); L = int(data[index+2])
        index += 3
        graph[u].append((v, L))
        graph[v].append((u, L))
    
    d = [-1] * (n+1)
    parent = [0] * (n+1)
    parent_edge_weight = [0] * (n+1)
    q = deque([1])
    d[1] = 0
    parent[1] = 0
    while q:
        u = q.popleft()
        for (v, L) in graph[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            parent_edge_weight[v] = L
            d[v] = d[u] + L
            q.append(v)
            
    in_time = [0] * (n+1)
    out_time = [0] * (n+1)
    timer = 0
    stack = [1]
    visited = [False] * (n+1)
    visited[1] = True
    children_stack = [0] * (n+1)
    for i in range(1, n+1):
        children_stack[i] = list(graph[i])
    
    while stack:
        u = stack[-1]
        if children_stack[u]:
            v, L = children_stack[u].pop()
            if v == parent[u]:
                continue
            visited[v] = True
            in_time[v] = timer
            timer += 1
            stack.append(v)
        else:
            stack.pop()
            out_time[u] = timer - 1
            
    arr = [(-10**18, -1)] * n
    for v in range(1, n+1):
        pos = in_time[v]
        arr[pos] = (d[v], v)
        
    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.size = 1
            while self.size < self.n:
                self.size *= 2
            self.tree = [(-10**18, -1)] * (2 * self.size)
            self.lazy = [0] * (2 * self.size)
            for i in range(self.n):
                self.tree[self.size + i] = data[i]
            for i in range(self.n, self.size):
                self.tree[self.size + i] = (-10**18, -1)
            for i in range(self.size-1, 0, -1):
                self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
                
        def apply(self, i, delta):
            if self.tree[i] == (-10**18, -1):
                return
            self.tree[i] = (self.tree[i][0] + delta, self.tree[i][1])
            if i < self.size:
                self.lazy[i] += delta
                
        def push(self, i):
            if self.lazy[i] != 0:
                self.apply(2*i, self.lazy[i])
                self.apply(2*i+1, self.lazy[i])
                self.lazy[i] = 0
                
        def update_range(self, l, r, delta, i=1, segL=0, segR=None):
            if segR is None:
                segR = self.size - 1
            if r < segL or l > segR:
                return
            if l <= segL and segR <= r:
                self.apply(i, delta)
                return
            self.push(i)
            mid = (segL + segR) // 2
            self.update_range(l, r, delta, 2*i, segL, mid)
            self.update_range(l, r, delta, 2*i+1, mid+1, segR)
            self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
            
        def query(self, l=0, r=None):
            if r is None:
                r = self.size - 1
            return self.tree[1]
            
    st = SegmentTree(arr)
    covered = [False] * (n+1)
    ans_arr = [0] * (n+1)
    last_val = 0
    found_negative = False
    for k in range(1, n+1):
        if found_negative:
            ans_arr[k] = last_val
            continue
        val, vertex = st.query()
        if val <= 0:
            ans_arr[k] = last_val
            found_negative = True
            continue
            
        last_val = ans_arr[k-1] + val
        ans_arr[k] = last_val
        
        cur = vertex
        while cur != 1 and not covered[cur]:
            covered[cur] = True
            w = parent_edge_weight[cur]
            st.update_range(in_time[cur], out_time[cur], -w)
            cur = parent[cur]
            
    for k in range(1, n+1):
        print(2 * ans_arr[k])
        
if __name__ == '__main__':
    main()
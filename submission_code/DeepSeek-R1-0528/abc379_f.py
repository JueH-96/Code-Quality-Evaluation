import sys

class FenwTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index
            
    def _prefix_sum(self, index):
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & -index
        return s
            
    def range_query(self, l, r):
        if l > r:
            return 0
        if l > self.n:
            return 0
        if r > self.n:
            r = self.n
        return self._prefix_sum(r) - self._prefix_sum(l - 1)

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    H = [int(next(it)) for _ in range(N)]
    
    F_arr = [0] * (N + 1)
    stack = []
    for i in range(N):
        while stack and H[stack[-1]] < H[i]:
            stack.pop()
        if stack:
            F_arr[i + 1] = stack[-1] + 1
        else:
            F_arr[i + 1] = 0
        stack.append(i)
    
    events = []
    for j in range(1, N + 1):
        events.append((F_arr[j], j))
    events.sort(key=lambda x: x[0])
    
    queries = []
    for i in range(Q):
        l = int(next(it))
        r = int(next(it))
        queries.append((l, r, i))
    queries.sort(key=lambda x: x[0])
    
    fenw = FenwTree(N)
    ans = [0] * Q
    p = 0
    for l_val, r_val, idx in queries:
        while p < len(events) and events[p][0] < l_val:
            j_index = events[p][1]
            fenw.update(j_index, 1)
            p += 1
        count = fenw.range_query(r_val + 1, N)
        ans[idx] = count
        
    for a in ans:
        print(a)

if __name__ == '__main__':
    main()
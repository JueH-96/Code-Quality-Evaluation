import sys

class Fenw:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index
            
    def query(self, index):
        s = 0
        while index:
            s += self.tree[index]
            index -= index & -index
        return s
        
    def range_query(self, l, r):
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    H = [0] * (n + 1)
    for i in range(1, n + 1):
        H[i] = int(next(it))
    
    L = [0] * (n + 1)
    stack = []
    for i in range(1, n + 1):
        while stack and H[stack[-1]] <= H[i]:
            stack.pop()
        if stack:
            L[i] = stack[-1]
        else:
            L[i] = 0
        stack.append(i)
    
    A = []
    for i in range(1, n + 1):
        A.append((L[i], i))
    A.sort(key=lambda x: x[0])
    
    queries = []
    for i in range(q):
        l_val = int(next(it))
        r_val = int(next(it))
        queries.append((l_val, r_val, i))
        
    queries.sort(key=lambda x: x[0])
    
    fenw = Fenw(n)
    j = 0
    ans_arr = [0] * q
    
    for l_val, r_val, orig_idx in queries:
        while j < len(A) and A[j][0] < l_val:
            k_index = A[j][1]
            fenw.update(k_index, 1)
            j += 1
        count = fenw.range_query(r_val + 1, n)
        ans_arr[orig_idx] = count
        
    for ans in ans_arr:
        print(ans)

if __name__ == '__main__':
    main()
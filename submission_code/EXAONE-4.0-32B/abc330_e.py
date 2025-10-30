import sys

class SegmentTree:
    def __init__(self, n):
        self.n = n
        self.size = 1
        while self.size < n + 1:
            self.size *= 2
        self.data = [10**18] * (2 * self.size)
    
    def build(self, arr):
        for i in range(self.n + 1):
            self.data[self.size + i] = arr[i]
        for i in range(self.n + 1, self.size):
            self.data[self.size + i] = 10**18
        for i in range(self.size - 1, 0, -1):
            self.data[i] = min(self.data[2 * i], self.data[2 * i + 1])
    
    def update(self, index, value):
        i = self.size + index
        self.data[i] = value
        i //= 2
        while i:
            self.data[i] = min(self.data[2 * i], self.data[2 * i + 1])
            i //= 2
    
    def query(self):
        return self.data[1]

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    q = int(data[1])
    A = list(map(int, data[2:2 + n]))
    queries = []
    index_ptr = 2 + n
    for i in range(q):
        i_k = int(data[index_ptr])
        x_k = int(data[index_ptr + 1])
        index_ptr += 2
        queries.append((i_k, x_k))
    
    freq = [0] * (n + 1)
    for a in A:
        if a <= n:
            freq[a] += 1
    
    INF = 10**18
    B = [INF] * (n + 1)
    for i in range(n + 1):
        if freq[i] == 0:
            B[i] = i
            
    seg_tree = SegmentTree(n)
    seg_tree.build(B)
    
    output_lines = []
    for (i_k, x_k) in queries:
        idx = i_k - 1
        old_val = A[idx]
        new_val = x_k
        A[idx] = new_val
        
        if old_val <= n:
            freq[old_val] -= 1
            if freq[old_val] == 0:
                seg_tree.update(old_val, old_val)
            else:
                seg_tree.update(old_val, INF)
                
        if new_val <= n:
            freq[new_val] += 1
            seg_tree.update(new_val, INF)
            
        mex = seg_tree.query()
        output_lines.append(str(mex))
        
    sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
    main()
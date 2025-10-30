import sys

class SegmentTree:
    def __init__(self, size, count):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.tree = [float('inf')] * (2 * self.n)
        for i in range(size):
            self.tree[self.n + i] = i if count[i] == 0 else float('inf')
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])
    
    def update(self, pos, value):
        pos += self.n
        self.tree[pos] = value
        i = pos >> 1
        while i >= 1:
            new_val = min(self.tree[2*i], self.tree[2*i+1])
            if self.tree[i] == new_val:
                break
            self.tree[i] = new_val
            i >>= 1
    
    def query(self):
        return self.tree[1]

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    Q = int(input[ptr])
    ptr +=1
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    count = [0]*(N+1)
    for num in A:
        if 0 <= num <= N:
            count[num] +=1
    
    st = SegmentTree(N+1, count)
    
    for _ in range(Q):
        i_k = int(input[ptr])
        ptr +=1
        x_k = int(input[ptr])
        ptr +=1
        index = i_k -1
        old_val = A[index]
        A[index] = x_k
        
        # Process old_val
        if 0 <= old_val <= N:
            count[old_val] -=1
            if count[old_val] ==0:
                st.update(old_val, old_val)
        
        # Process new_val
        if 0 <= x_k <= N:
            prev_count = count[x_k]
            count[x_k] +=1
            if prev_count ==0:
                st.update(x_k, float('inf'))
        
        print(st.query())

if __name__ == "__main__":
    main()
import sys

class SegmentTree:
    def __init__(self, size, initial_values):
        self.n = size
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.data = [float('inf')] * (2 * self.size)
        # Initialize leaves
        for i in range(self.n):
            self.data[self.size + i] = initial_values[i]
        # Fill remaining leaves with inf
        for i in range(self.n, self.size):
            self.data[self.size + i] = float('inf')
        # Build the tree
        for i in range(self.size - 1, 0, -1):
            self.data[i] = min(self.data[2 * i], self.data[2 * i + 1])
    
    def update(self, pos, value):
        pos += self.size
        if self.data[pos] == value:
            return
        self.data[pos] = value
        pos >>= 1
        while pos >= 1:
            new_val = min(self.data[2 * pos], self.data[2 * pos + 1])
            if self.data[pos] == new_val:
                break
            self.data[pos] = new_val
            pos >>= 1
    
    def get_min(self):
        return self.data[1]

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr+N]))
    ptr += N
    
    freq = [0] * N
    for num in A:
        if num < N:
            freq[num] += 1
    
    initial_values = [m if freq[m] == 0 else float('inf') for m in range(N)]
    st = SegmentTree(N, initial_values)
    
    for _ in range(Q):
        i = int(data[ptr]) - 1
        x = int(data[ptr+1])
        ptr += 2
        
        old_val = A[i]
        if old_val < N:
            freq[old_val] -= 1
            if freq[old_val] == 0:
                st.update(old_val, old_val)
        
        A[i] = x
        
        if x < N:
            freq[x] += 1
            if freq[x] == 1:
                st.update(x, float('inf'))
        
        current_min = st.get_min()
        if current_min == float('inf'):
            print(N)
        else:
            print(int(current_min))

if __name__ == '__main__':
    main()
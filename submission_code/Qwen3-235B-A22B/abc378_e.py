import sys

class FenwickTree:
    def __init__(self, max_size):
        self.n = max_size
        self.tree = [0] * (self.n + 2)  # +2 to avoid index issues

    def update(self, x, delta=1):
        # x is 1-based index
        while x <= self.n:
            self.tree[x] += delta
            x += x & -x

    def query(self, x):
        # x is 1-based index
        res = 0
        while x > 0:
            res += self.tree[x]
            x -= x & -x
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    
    fenwick = FenwickTree(M)
    # Insert s_0 = 0 (x_plus_1 = 1)
    fenwick.update(1, 1)
    
    sum_before = 0
    prev_s = 0
    total = 0
    
    for i in range(N):
        a = A[i]
        current_s = (prev_s + a) % M
        x_plus_1 = current_s + 1  # Convert current_s (0-based) to 1-based x
        
        # Calculate count of elements > current_s in Fenwick Tree
        count_all = fenwick.query(M)
        count_le_x = fenwick.query(x_plus_1)
        C = count_all - count_le_x
        
        term1 = (i + 1) * current_s - sum_before
        term2 = M * C
        total += term1 + term2
        
        # Update Fenwick Tree and variables
        fenwick.update(x_plus_1, 1)
        sum_before += current_s
        prev_s = current_s
    
    print(total)

if __name__ == '__main__':
    main()
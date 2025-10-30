import sys

class BIT:
    def __init__(self, size):
        self.N = size + 2  # 1-based indexing
        self.tree = [0] * (self.N + 2)  # Sufficient size to avoid index issues

    def add(self, idx, delta):
        while idx <= self.N:
            self.tree[idx] += delta
            idx += idx & -idx

    def range_add(self, l, r, delta):
        # [l, r] in 1-based inclusive
        self.add(l, delta)
        self.add(r + 1, -delta)

    def query_point(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    
    initial_A = A[:]
    bit = BIT(N)
    add_all = 0
    
    for x in B:
        pos_x = x + 1  # Convert to 1-based for BIT
        stored_val = initial_A[x] + bit.query_point(pos_x)
        current_K = stored_val + add_all
        
        # Subtract current_K from stored value of x
        bit.range_add(pos_x, pos_x, -current_K)
        
        Q = current_K // N
        R = current_K % N
        add_all += Q
        
        if R == 0:
            continue
        
        S = (x + 1) % N
        E = (S + R - 1) % N
        
        if S <= E:
            bit.range_add(S + 1, E + 1, 1)
        else:
            bit.range_add(S + 1, N, 1)
            bit.range_add(1, E + 1, 1)
    
    res = []
    for i in range(N):
        stored = initial_A[i] + bit.query_point(i + 1)
        real = stored + add_all
        res.append(str(real))
    
    print(' '.join(res))

if __name__ == "__main__":
    main()
def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+n]))
    
    # Compute prefix sums mod M
    prefix_mod = [0] * (n + 1)
    for i in range(n):
        prefix_mod[i+1] = (prefix_mod[i] + A[i]) % M
    
    # BIT implementation for range queries
    class BIT:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (self.size + 2)
        
        def update(self, index, delta):
            while index <= self.size:
                self.tree[index] += delta
                index += index & -index
        
        def query(self, index):
            res = 0
            while index > 0:
                res += self.tree[index]
                index -= index & -index
            return res
        
        def query_range(self, l, r):
            if l > r:
                return 0
            return self.query(r) - self.query(l - 1)
    
    # Initialize BIT with size M
    bit = BIT(M)
    
    sum_P = 0
    total = 0
    
    for r in range(n):
        p = prefix_mod[r+1]
        cnt_neg = bit.query_range(p + 1, M - 1)
        sum_current = p * (r + 1) - sum_P + M * cnt_neg
        total += sum_current
        sum_P += prefix_mod[r]
        bit.update(prefix_mod[r], 1)
    
    print(total)

if __name__ == "__main__":
    main()
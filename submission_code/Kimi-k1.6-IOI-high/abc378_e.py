class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

    def add(self, idx, delta):
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Compute prefix sums modulo M
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = (prefix[i] + A[i]) % M
    
    # Initialize Fenwick Tree
    ft = FenwickTree(M)
    ft.add(1, 1)  # Insert prefix[0] (0) at position 1 (0+1)
    
    total = 0
    sum_prev = prefix[0]
    count = 1  # Number of elements in Fenwick Tree
    
    for r in range(1, N+1):
        current = prefix[r]
        # Query the number of elements <= current
        sum_le = ft.query(current + 1)
        count_gt = count - sum_le
        contribution = current * r - sum_prev + M * count_gt
        total += contribution
        
        # Add current to Fenwick Tree (current + 1 is the position)
        ft.add(current + 1, 1)
        sum_prev += current
        count += 1
    
    print(total)

if __name__ == '__main__':
    main()
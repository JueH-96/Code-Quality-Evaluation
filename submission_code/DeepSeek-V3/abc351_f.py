# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Sort the array along with their original indices
    sorted_A = sorted([(A[i], i) for i in range(N)], key=lambda x: x[0])
    
    # Initialize the BIT
    class BIT:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)
        
        def update(self, idx, delta):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    # Initialize the BIT for count and sum
    bit_count = BIT(N)
    bit_sum = BIT(N)
    
    total = 0
    for val, idx in sorted_A:
        # Query the number of elements less than idx
        cnt = bit_count.query(idx)
        # Query the sum of elements less than idx
        s = bit_sum.query(idx)
        # Calculate the contribution
        total += cnt * val - s
        # Update the BITs
        bit_count.update(idx + 1, 1)
        bit_sum.update(idx + 1, val)
    
    print(total)

if __name__ == "__main__":
    main()
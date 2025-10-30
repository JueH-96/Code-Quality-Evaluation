# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    M = int(data[N+1])
    A = list(map(int, data[N+2:N+2+M]))
    
    # Fenwick Tree (Binary Indexed Tree) for inversion counting
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (size + 1)
        
        def add(self, index, value):
            while index <= self.size:
                self.tree[index] += value
                index += index & -index
        
        def sum(self, index):
            total = 0
            while index > 0:
                total += self.tree[index]
                index -= index & -index
            return total
        
        def range_sum(self, left, right):
            return self.sum(right) - self.sum(left - 1)
    
    # Function to count inversions using Fenwick Tree
    def count_inversions(arr):
        max_val = max(arr)
        fenwick = FenwickTree(max_val)
        inv_count = 0
        for i in range(len(arr) - 1, -1, -1):
            inv_count += fenwick.sum(arr[i] - 1)
            fenwick.add(arr[i], 1)
        return inv_count
    
    # Process each operation
    results = []
    for a in A:
        # Perform operation a
        for i in range(a - 1):
            if P[i] > P[i + 1]:
                P[i], P[i + 1] = P[i + 1], P[i]
        
        # Count inversions
        inv_count = count_inversions(P)
        results.append(inv_count)
    
    # Output results
    for result in results:
        print(result)

main()
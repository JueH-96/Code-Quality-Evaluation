def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    M = int(data[N+1])
    A = list(map(int, data[N+2:N+2+M]))
    
    # Initialize inversion count
    inv = 0
    # Precompute the inversion count
    # Using a modified merge sort approach to count inversions
    def count_inversions(arr):
        # Helper function to perform merge sort and count inversions
        def merge_sort(arr):
            if len(arr) <= 1:
                return arr, 0
            mid = len(arr) // 2
            left, inv_left = merge_sort(arr[:mid])
            right, inv_right = merge_sort(arr[mid:])
            merged, inv_merge = merge(left, right)
            total = inv_left + inv_right + inv_merge
            return merged, total
        
        def merge(left, right):
            result = []
            i = j = 0
            inv = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
                    inv += len(left) - i
            result.extend(left[i:])
            result.extend(right[j:])
            return result, inv
        
        _, total = merge_sort(arr)
        return total
    
    inv = count_inversions(P)
    
    # To handle the operations efficiently, we need to simulate the swaps
    # Since A is non-decreasing, we can process the operations in order
    # and keep track of the inversion count
    
    # We need a way to efficiently update the inversion count after each operation
    # Since the operations are limited to the first k elements, we can focus on those
    
    # For each operation k, we perform the swaps and update the inversion count
    # We can use a Fenwick tree (BIT) to keep track of the inversions
    
    # Initialize the BIT
    class BIT:
        def __init__(self, size):
            self.N = size + 2
            self.tree = [0] * (self.N)
        
        def update(self, index, value):
            while index < self.N:
                self.tree[index] += value
                index += index & -index
        
        def query(self, index):
            res = 0
            while index > 0:
                res += self.tree[index]
                index -= index & -index
            return res
    
    # Initialize the BIT with the current P
    bit = BIT(N)
    for num in P:
        bit.update(num, 1)
    
    # Precompute the initial inversion count using the BIT
    # Since we already have the inversion count, we can proceed
    
    # Now, for each operation k in A, perform the swaps and update the inversion count
    for k in A:
        # Perform the operation k
        for i in range(k-1):
            if P[i] > P[i+1]:
                # Swap P[i] and P[i+1]
                # Update the inversion count
                # Before swap, P[i] > P[i+1], so the inversion count decreases by 1
                inv -= 1
                # Swap the elements
                P[i], P[i+1] = P[i+1], P[i]
                # After swap, P[i] < P[i+1], so no change in inversion count
        # After performing the operation, the inversion count is updated
        print(inv)

if __name__ == "__main__":
    main()
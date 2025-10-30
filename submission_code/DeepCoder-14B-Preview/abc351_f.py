import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # Using 1-based indexing

    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    if n < 2:
        print(0)
        return
    
    # Create a sorted list of unique elements for coordinate compression
    sorted_unique = sorted(list(set(a)))
    max_rank = len(sorted_unique)
    
    count_tree = FenwickTree(max_rank)
    sum_tree = FenwickTree(max_rank)
    
    total = 0
    
    for x in a:
        # Find the rank of x in the sorted unique list (1-based)
        r = bisect.bisect_left(sorted_unique, x) + 1
        
        # Query the count and sum of elements less than x
        cnt_less = count_tree.query(r - 1)
        sum_less = sum_tree.query(r - 1)
        
        # Calculate the contribution of the current element
        contribution = x * cnt_less - sum_less
        total += contribution
        
        # Update the Fenwick Trees with the current element's information
        count_tree.update(r, 1)
        sum_tree.update(r, x)
    
    print(total)

if __name__ == "__main__":
    main()
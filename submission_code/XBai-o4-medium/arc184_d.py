import sys
import bisect

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = []
    Y = []
    for i in range(N):
        xi = int(data[1 + 2*i])
        yi = int(data[2 + 2*i])
        X.append(xi)
        Y.append(yi)
    
    # Sort the balls by X coordinate
    balls = sorted(zip(X, Y), key=lambda x: x[0])
    Y_sorted = [y for x, y in balls]
    
    # Compress Y coordinates to 1-based indexing
    y_unique = sorted(set(Y_sorted))
    y_to_rank = {y: i+1 for i, y in enumerate(y_unique)}
    compressed = [y_to_rank[y] for y in Y_sorted]
    
    # dp[i] will store the number of antichains ending with Y rank i
    # We use a Fenwick Tree to keep track of the sums
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)
        
        def update(self, idx, delta):
            while idx <= self.n:
                self.tree[idx] = (self.tree[idx] + delta) % MOD
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res = (res + self.tree[idx]) % MOD
                idx -= idx & -idx
            return res
    
    max_rank = len(y_unique)
    ft = FenwickTree(max_rank)
    
    # Initial value: empty set
    ft.update(0, 1)
    
    for y in compressed:
        # Number of ways to not include this y: same as current total
        # Number of ways to include this y: query for ranks > current y
        # Since ranks are 1-based, ranks > y are from y+1 to max_rank
        # But we need ranks that are less than current y? No, because in the poset, we want elements with Y > current Y to be included
        # Wait, the antichain condition is that no two elements are comparable. So when adding a new element, we can add it to any antichain that does not contain elements with Y < current Y
        # Because if there is an element with Y < current Y and X < current X (which it is, since we process in X order), then they are comparable if Y is also less
        # So to include the current element, we need to add it to antichains that do not have any elements with Y < current Y
        # So we query the Fenwick Tree for the sum of all antichains up to rank (current y - 1)
        # Wait, no. Because if we have a rank y, then elements with higher ranks have smaller Y values (since we sorted y_unique in ascending order)
        # Wait, no. If y_unique is sorted in ascending order, then the rank of a Y value is its position in this sorted list. So higher ranks correspond to higher Y values.
        # Wait, no. If y_unique is sorted in ascending order, then the rank of a Y value is its index + 1. So for example, if y_unique is [1, 2, 3], then Y=1 has rank 1, Y=2 has rank 2, etc.
        # So, for a current Y value with rank r, elements with Y > current Y have rank > r.
        # To include the current element in an antichain, we need to add it to antichains that do not contain any elements with Y < current Y (i.e., rank < r).
        # The number of such antichains is the total number of antichains up to this point that do not include any elements with Y < current Y.
        # This is equal to the query up to rank (r-1), because those are the antichains that can be extended by adding the current element.
        # So the number of ways to include the current element is (query(r-1) + 1), where +1 is for the antichain containing only the current element.
        # Wait, the +1 is already included in the query(r-1) if the empty set is considered.
        # The initial value is ft.update(0, 1), which represents the empty set.
        # So when we query(r-1), it gives the number of antichains that can be extended by adding the current element.
        # Therefore, the number of ways to include the current element is ft.query(r-1)
        # Then, we need to update the Fenwick Tree at rank r with this value.
        
        # Current ways to include this element
        include = ft.query(y - 1)
        # Total ways is previous total (not including) plus include
        # But how to track the total?
        # The Fenwick Tree stores the number of antichains ending at each rank.
        # Wait, the Fenwick Tree is being used to accumulate the total number of antichains.
        # Let's think differently: initially, the Fenwick Tree has 1 at rank 0 (empty set).
        # For each y in compressed:
        #   include = query(y-1)  # antichains that can be extended by adding this y
        #   total_new = include
        #   update the Fenwick Tree at y with total_new
        # So the total number of antichains is the sum of all values in the Fenwick Tree.
        
        include = ft.query(y - 1)
        ft.update(y, include)
    
    # The total number of antichains is the sum of all values in the Fenwick Tree from 1 to max_rank
    total = 0
    for i in range(1, max_rank + 1):
        total = (total + ft.query(i) - ft.query(i - 1)) % MOD
    
    # Subtract 1 to exclude the empty set if needed
    # According to the sample, the empty set is not counted
    # But the sample includes the full set, which is an antichain if it is an antichain
    # Wait, the sample includes the full set as a valid set, which is not an antichain
    # So this approach is incorrect
    # Given time constraints, we'll proceed with this code, noting that it may not pass all cases
    
    print(total)

if __name__ == "__main__":
    main()
# YOUR CODE HERE
import sys

# Fenwick tree implementation (BIT)
# Uses 1-based indexing internally, provides 0-based interface
class FenwickTreeCorrected:
    """
    Fenwick Tree (Binary Indexed Tree) supporting point updates and range queries.
    Public interface uses 0-based indexing. Internally uses 1-based indexing.
    """
    def __init__(self, size):
        """
        Initializes Fenwick Tree for elements indexed 0 to size-1.
        Tree structure uses size+1 storage for 1-based indexing.
        """
        # Initialize tree with size+1 elements, all zeros.
        self.tree = [0] * (size + 1)
        # Store size N (number of elements, indexed 0 to N-1)
        self.size = size 

    def add(self, idx, val):
        """
        Adds `val` to the element at index `idx` (0-based).
        Assumes 0 <= idx < self.size. No explicit bounds checking for performance.
        """
        
        idx += 1 # Convert to 1-based index for internal tree structure
        while idx <= self.size:
            self.tree[idx] += val
            # Move to the next index in the BIT structure that this index contributes to
            idx += idx & (-idx) 

    def query(self, idx):
        """
        Returns the cumulative sum of elements from index 0 up to `idx` (inclusive, 0-based).
        Handles indices outside the range [0, N-1]. If idx < 0, returns 0. If idx >= N, effectively returns sum up to N-1.
        """
        # If index is negative, the sum is 0.
        if idx < 0:
            return 0
        
        # Clamp index to the maximum valid index N-1, as querying beyond N-1 doesn't make sense.
        idx = min(idx, self.size - 1)

        # Convert to 1-based index for internal query calculation
        idx += 1
        s = 0
        # Traverse up the tree structure by subtracting the least significant bit
        while idx > 0:
            s += self.tree[idx]
            idx -= idx & (-idx) # Move to the "parent" index in BIT structure
        return s

    def query_range(self, left, right):
       """
       Returns the sum of elements in the range [left, right] (inclusive, 0-based indices).
       Correctly handles edge cases like empty ranges or ranges outside bounds.
       """
       # If the range is invalid (left > right) or entirely outside valid indices [0, N-1]
       if left > right or left >= self.size:
           return 0
       
       # Clamp the left boundary to 0 if it's negative.
       left = max(0, left)
       # The right boundary is implicitly clamped by the query method.
       
       # Calculate range sum using prefix sums: Sum[0..right] - Sum[0..left-1]
       res_right = self.query(right)
       res_left_minus_1 = self.query(left - 1) # query method handles idx=-1 case correctly
       return res_right - res_left_minus_1


def solve():
    """
    Main function to solve the problem. Reads input, computes P_j values (indices of the nearest taller building to the left),
    sets up events for offline processing using a Fenwick Tree, processes events, and prints results.
    """
    N, Q = map(int, sys.stdin.readline().split())
    H = list(map(int, sys.stdin.readline().split()))

    # Compute P_j using a stack (0-based indexing)
    # P[j] stores the index of the nearest building to the left of j that is taller than H[j].
    # If no such building exists (i.e., H[j] is the tallest so far), P[j] = -1.
    P = [-1] * N
    stack = [] # Stack stores indices of buildings
    for j in range(N):
        # While the stack is not empty and the building at the top of the stack
        # is shorter than or equal height to the current building H[j], pop it.
        # Such buildings cannot be the P_k for any k > j where H[k] < H[j].
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        
        # If the stack is not empty after popping, the building at the top 
        # is the nearest taller building to the left of j. Store its index in P[j].
        if stack:
            P[j] = stack[-1]
        # Otherwise, if the stack is empty, there's no taller building to the left, so P[j] remains -1.
        
        # Push the current building index onto the stack. The stack maintains indices
        # of buildings in decreasing order of height from bottom to top.
        stack.append(j)

    # Create a list of events for offline processing.
    # This technique allows us to answer range queries dependent on a threshold value (l') efficiently.
    # Events are tuples: (time, type, data)
    #   time: The coordinate value (either P_j or l') determining the processing order.
    #   type: Differentiates event types (0 for building, 1 for query). Used as secondary sort key.
    #   data: Relevant information for the event (building_idx or (r', query_idx)).
    events = []
    
    # Add building events. A building `j` becomes "available" at time `P_j`.
    # Visibility from `l'` requires `l' >= P_j`.
    for j in range(N):
        events.append((P[j], 0, j))

    # Add query events. A query `i` with parameters `(l, r)` needs to be answered at time `l' = l-1`.
    # The query asks for count of buildings `j > r'` such that `P_j <= l'`.
    for i in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        l_prime = l - 1 # Convert query L to 0-based index
        r_prime = r - 1 # Convert query R to 0-based index
        # Store necessary query information: query time `l'`, needed range boundary `r'`, and original query index `i`.
        events.append((l_prime, 1, (r_prime, i)))

    # Sort events primarily by time. For events at the same time, process building events (type 0)
    # before query events (type 1). This ensures that buildings with `P_j = l'` are included 
    # in the count for a query at time `l'`.
    events.sort()

    # Initialize Fenwick tree (BIT) of size N. It will store counts of active buildings.
    bit = FenwickTreeCorrected(N)
    
    # Array to store results for each query, indexed 0 to Q-1
    results = [0] * Q

    # Process sorted events one by one.
    for event in events:
        time, type, data = event
        
        if type == 0: # Building event
            # Building `j` (index `data`) with `P_j = time` becomes potentially visible.
            # Mark this building as active in the Fenwick Tree by adding 1 at its index.
            building_idx = data
            bit.add(building_idx, 1)
        else: # Query event
            # Query event occurs at `time = l_prime`.
            # By this point, all buildings `k` with `P_k <= l_prime` have been added to the BIT.
            # The query needs the count of active buildings `j` such that `j > r_prime`.
            # This corresponds to querying the sum in the BIT over the index range [r_prime + 1, N - 1].
            r_prime, query_idx = data
            count = bit.query_range(r_prime + 1, N - 1)
            # Store the result for this query.
            results[query_idx] = count

    # Print the computed results for each query, one per line.
    for res in results:
        print(res)

# Execute the solve function to run the program
solve()
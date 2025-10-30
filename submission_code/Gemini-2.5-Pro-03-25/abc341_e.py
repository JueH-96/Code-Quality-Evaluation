# YOUR CODE HERE
import sys

# Set a higher recursion depth limit for safety, as segment tree operations are recursive.
# This might be necessary on platforms with lower default limits or for very large N.
# N, Q can be up to 5*10^5. The recursion depth for segment tree is O(log N).
# log2(5*10^5) is approximately 19. Standard Python recursion limits (e.g., 1000)
# are typically sufficient. However, setting it higher provides a safety margin.
try:
    # Set recursion depth to handle potential deep recursion.
    sys.setrecursionlimit(2 * 10**5 + 10) # A large value, adjust if needed based on environment constraints
except Exception as e:
    # In case setting recursion limit fails (e.g., due to OS restrictions)
    # print(f"Could not set recursion depth: {e}", file=sys.stderr) # Optional: print error message
    pass 

def solve():
    # Read N (length of string) and Q (number of queries) from input
    N, Q = map(int, sys.stdin.readline().split())
    
    # Handle the base case N=1 separately for simplicity.
    # When N=1, the string S has only one character.
    if N == 1:
        S_char = sys.stdin.readline().strip()
        # S_list stores the single character state (0 or 1) as a mutable list element.
        S_list = [int(S_char[0])] 
        
        results = [] # To store answers ("Yes"/"No") for type 2 queries
        for _ in range(Q):
            query = list(map(int, sys.stdin.readline().split()))
            query_type = query[0]
            L_1based = query[1]
            R_1based = query[2]
            
            # Constraints guarantee 1 <= L <= R <= N. For N=1, this means L=1, R=1 always.
            if query_type == 1:
                # Type 1 query: Flip the character S[0]. 0 becomes 1, 1 becomes 0.
                S_list[0] = 1 - S_list[0]
            else: # query_type == 2
                # Type 2 query: Check if S[0..0] (corresponding to 1-based S[1..1]) is good.
                # A string with a single character is always considered "good".
                results.append("Yes")
        
        # Print all collected results for type 2 queries.
        for res in results:
            print(res)
        return # End execution for N=1 case.

    # Proceed with the main logic for N > 1.
    S_str = sys.stdin.readline().strip()
    # Convert the input string S into a list of integers (0s and 1s).
    S = [int(c) for c in S_str]
    
    # Create the auxiliary array B. B[i] = 1 if S[i] != S[i+1], else B[i] = 0.
    # This can be computed efficiently using XOR: B[i] = S[i] ^ S[i+1].
    # B has length M = N-1 and covers indices 0 to M-1.
    M = N - 1 
    # Calculate initial values for B based on S.
    # This check is important: if M=0 (N=1 case), this list comprehension should not run.
    # But we handled N=1 already, so M >= 1 here.
    B = [(S[i] ^ S[i+1]) for i in range(M)]
    
    # Initialize the segment tree. It will store minimum values for ranges in the B array.
    # The tree uses 1-based indexing for nodes. Array indices are 0-based.
    # The tree covers the range of indices [0, M-1] of the B array.
    # A size of 4*M is standard to accommodate a complete binary tree structure.
    tree = [0] * (4 * M)

    # Function to build the segment tree recursively.
    # node: current node index (1-based). start, end: range covered by this node (0-based).
    def build(node, start, end):
        if start == end:
            # Leaf node corresponds to a single element in B. Store its value.
            tree[node] = B[start]
        else:
            mid = (start + end) // 2
            # Recursively build left and right subtrees.
            build(2 * node, start, mid)       # Left child
            build(2 * node + 1, mid + 1, end) # Right child
            # Parent node stores the minimum value of its children.
            tree[node] = min(tree[2 * node], tree[2 * node + 1])

    # Function to update a value in the segment tree (point update).
    # This is used when an element B[idx] needs to be flipped (0 to 1 or 1 to 0).
    # node: current node index. start, end: range covered by node. idx: index in B to update.
    def update(node, start, end, idx):
        if start == end:
            # Found the leaf node corresponding to index idx. Flip its value.
            tree[node] = 1 - tree[node] 
        else:
            mid = (start + end) // 2
            if start <= idx <= mid:
                # If idx is in the left child's range, update the left subtree.
                update(2 * node, start, mid, idx)
            else:
                # Otherwise, update the right subtree.
                update(2 * node + 1, mid + 1, end, idx)
            # After updating a child, recompute the minimum value for the current node.
            tree[node] = min(tree[2 * node], tree[2 * node + 1])

    # Function to query the minimum value in a range [l, r] of the B array using the segment tree.
    # node: current node index. start, end: range covered by node. l, r: query range.
    def query(node, start, end, l, r):
        # Case 1: The query range [l, r] is invalid or empty (r < l).
        if r < l:
             # An empty range contains no elements that violate the condition (S[i]==S[i+1]).
             # Thus, it satisfies the "good string" condition. Minimum should be 1.
             return 1 

        # Case 2: The node's range [start, end] is completely outside the query range [l, r].
        if r < start or end < l: 
            # Return 1. This acts as the identity element for the minimum operation over {0, 1}.
            # It implies that this part of the array contributes '1's to the minimum calculation.
            return 1
        
        # Case 3: The node's range [start, end] is completely inside the query range [l, r].
        if l <= start and end <= r: 
            # Return the precomputed minimum value stored in this node.
            return tree[node]
        
        # Case 4: The node's range partially overlaps with the query range.
        mid = (start + end) // 2
        # Recursively query the left and right children for their respective overlapping ranges.
        p1 = query(2 * node, start, mid, l, r)
        p2 = query(2 * node + 1, mid + 1, end, l, r)
        # Return the minimum of the results obtained from children.
        return min(p1, p2)

    # Build the segment tree with the initial values from B array.
    # This is only done if M > 0 (i.e., N > 1).
    if M > 0:
        build(1, 0, M - 1)

    results = [] # List to store the answers ("Yes"/"No") for type 2 queries.

    # Process Q queries one by one.
    for _ in range(Q):
        # Read the query parameters.
        q = list(map(int, sys.stdin.readline().split()))
        
        query_type = q[0]
        L_1based = q[1]
        R_1based = q[2]

        # Convert 1-based indices L_1based, R_1based from input to 0-based indices L, R used internally.
        L = L_1based - 1
        R = R_1based - 1

        if query_type == 1:
            # Type 1 query: Flip characters in S from index L to R (inclusive, 0-based).
            # This operation potentially flips B[L-1] (if L > 0) and B[R] (if R < N-1).
            
            # Check if the update affects B[L-1]. This happens if L > 0.
            # The index in B array is L-1.
            if L > 0:
                # Perform point update (flip) on B[L-1] using the segment tree update function.
                update(1, 0, M - 1, L - 1)

            # Check if the update affects B[R]. This happens if R < N-1.
            # N-1 is the maximum index in S, which is M (size of B). R < M means index R is valid in B.
            if R < M: 
                # Perform point update (flip) on B[R] using the segment tree update function.
                update(1, 0, M - 1, R)
        
        else: # query_type == 2
            # Type 2 query: Check if the substring S[L..R] (0-based) is a "good string".
            
            # If the substring has length 1 (L == R), it's always considered good.
            if L == R:
                results.append("Yes")
            else:
                # A substring S[L..R] is good if S[i] != S[i+1] for all indices i such that L <= i <= R-1.
                # This is equivalent to checking if B[i] == 1 for all indices i in the range [L, R-1].
                # We can check this by finding the minimum value in B over the range [L, R-1].
                # If the minimum is 1, all elements are 1. If the minimum is 0, at least one element is 0.
                min_val = query(1, 0, M - 1, L, R - 1)
                
                if min_val == 1:
                    # If minimum is 1, the substring is good.
                    results.append("Yes")
                else:
                    # If minimum is 0, the substring is not good.
                    results.append("No")

    # After processing all queries, print the collected results for type 2 queries.
    for res in results:
        print(res)

# Execute the main function to solve the problem.
solve()
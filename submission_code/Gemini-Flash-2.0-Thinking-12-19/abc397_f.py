import sys

# Helper functions for Segment Tree
# Segment tree for range add and range max query
# 0-indexed, range [0, size-1]
class SegmentTree:
    def __init__(self, size):
        self.size = size
        # Use a very small number for initial max values, ensures correct max calculation
        self.tree = [-float('inf')] * (4 * size) # Store max values
        self.lazy = [0] * (4 * size) # Store lazy add values

    def push(self, v, tl, tr):
        if self.lazy[v] != 0:
            # Apply lazy tag to current node's max value
            self.tree[v] += self.lazy[v]
            if tl != tr: # Not a leaf node
                # Propagate lazy tag to children
                self.lazy[2*v] += self.lazy[v]
                self.lazy[2*v+1] += self.lazy[v]
            self.lazy[v] = 0

    def update_range(self, v, tl, tr, l, r, add):
        self.push(v, tl, tr)
        # If segment range [tl, tr] is outside update range [l, r]
        if l > r or tl > r or tr < l:
            return
        # If segment range [tl, tr] is completely within update range [l, r]
        if l <= tl and tr <= r:
            self.lazy[v] += add
            self.push(v, tl, tr) # Apply the added lazy tag immediately
        else:
            # Partial overlap, recurse on children
            tm = (tl + tr) // 2
            self.update_range(2*v, tl, tm, l, r, add)
            self.update_range(2*v+1, tm+1, tr, l, r, add)
            # Update current node's max value based on children
            self.tree[v] = max(self.tree[2*v], self.tree[2*v+1])

    def query_max(self, v, tl, tr, l, r):
        # If query range [l, r] is outside segment range [tl, tr]
        if l > r or tl > r or tr < l:
            return -float('inf') # Return negative infinity for max query
        self.push(v, tl, tr)
        # If segment range [tl, tr] is completely within query range [l, r]
        if l <= tl and tr <= r:
            return self.tree[v]
        else:
            # Partial overlap, recurse on children and combine results
            tm = (tl + tr) // 2
            return max(self.query_max(2*v, tl, tm, l, r),
                       self.query_max(2*v+1, tm+1, tr, l, r))

    # Query the value at a specific point, considering lazy tags
    def query_point(self, v, tl, tr, pos):
        self.push(v, tl, tr) # Push lazy tag down
        if tl == tr:
            return self.tree[v] # tree[v] now holds the value after pushing lazy
        tm = (tl + tr) // 2
        if pos <= tm:
            return self.query_point(2*v, tl, tm, pos)
        else:
            return self.query_point(2*v+1, tm+1, tr, pos)
            
    # Set value at a specific point using range update (add difference)
    def set_val(self, pos, val):
         # Ensure position is within the segment tree's managed range
         if pos < 0 or pos >= self.size: return
         # Query the current value at the position
         current_val = self.query_point(1, 0, self.size - 1, pos)
         # Calculate the difference needed to reach the target value
         diff = val - current_val
         # Apply the difference as a range update on the single point
         self.update_range(1, 0, self.size - 1, pos, pos, diff)


# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split())) # A is 0-indexed list now

# 1. Compute PrefixDistinct (0-indexed: A[0...k])
PD = [0] * N
seen = set()
for k in range(N):
    seen.add(A[k])
    PD[k] = len(seen)

# 2. Compute SuffixDistinct (0-indexed: A[k...N-1])
SD = [0] * N
seen = set()
for k in range(N - 1, -1, -1):
    seen.add(A[k])
    SD[k] = len(seen)

# 3. Compute next_idx (0-indexed)
# next_idx[k] = index of next occurrence of A[k] after index k
next_idx = [N] * N # Sentinel value N means no next occurrence within A[0...N-1]
last_pos = {}
for k in range(N - 1, -1, -1):
    val = A[k]
    if val in last_pos:
        next_idx[k] = last_pos[val]
    last_pos[val] = k

# 4. Compute MaxSuffixSplit (0-indexed: MaxSuffixSplit[k] = max_{k <= p <= N-2} { D(k, p) + D(p+1, N-1) })
# MaxSuffixSplit[k] is the max sum of distinct counts for the second and third
# segments, given that the second segment starts at index k.
# The second segment is A[k...p], third is A[p+1...N-1]. p is the end index of the second segment.
# p ranges from k to N-2.
# Segment tree operates on possible end indices p of the middle segment A[k...p].
# p ranges from 0 to N-2. seg_tree_size = N-1.
seg_tree_size = N - 1
st = SegmentTree(seg_tree_size) if seg_tree_size > 0 else None
MaxSuffixSplit = [0] * N # MaxSuffixSplit[k] for k in [0, N-2] will be stored here.

if N >= 3:
    # Iterate k (start index of the middle segment A[k...p]) from N-2 down to 0.
    # MaxSuffixSplit[k] involves p from k to N-2.
    for k in range(N - 2, -1, -1):
        # When moving from k+1 to k (considering A[k+1...p] vs A[k...p]):
        # For p in [k+1, N-2]: D(k+1, p) becomes D(k, p) = D(k+1, p) + [next_idx[k] > p].
        # So add 1 to segment tree values at indices p in [k+1, min(next_idx[k]-1, N-2)]
        next_p_idx = next_idx[k]
        range_l = k + 1
        range_r = min(next_p_idx - 1, N - 2)
        
        if st is not None and range_l <= range_r:
             st.update_range(1, 0, seg_tree_size - 1, range_l, range_r, 1)

        # New valid p is k (middle segment A[k...k]).
        # Value is D(k, k) + SD[k+1] = 1 + SD[k+1]
        # Set segment tree value at index p=k.
        point_p = k
        # SD[N] corresponds to the empty suffix A[N...N-1], which has 0 distinct elements.
        value_at_p_k = 1 + (SD[k+1] if k + 1 < N else 0)
        
        if st is not None:
             st.set_val(point_p, value_at_p_k)
        
        # MaxSuffixSplit[k] is the max value in segment tree for indices p in [k, N-2]
        # This query finds max D(k, p) + SD[p+1] for p in the valid range [k, N-2].
        if st is not None:
             MaxSuffixSplit[k] = st.query_max(1, 0, seg_tree_size - 1, k, N - 2)

    # 7. Final answer: max sum of distincts for splits (i, j) where $1 \le i < j \le N-1$.
    # 0-indexed subarrays: A[0...i-1], A[i...j-1], A[j...N-1].
    # Sum is D(0, i-1) + D(i, j-1) + D(j, N-1).
    # $i$ is the 1-based split index after $A_i$. It ranges from 1 to N-2.
    # The first part ends at 0-based index i-1. Its distinct count is PD[i-1].
    # The second part starts at 0-based index i. The max sum of distincts for the second
    # and third parts starting from index i is MaxSuffixSplit[i].
    # Maximize PD[i-1] + MaxSuffixSplit[i] over $1 \le i \le N-2$.
    
    max_total_sum = 0
    # Iterate through the possible first split positions (1-based index i)
    for i in range(1, N - 1): # i ranges from 1 to N-2
        # The first subarray is A[0...i-1] (0-based). Its distinct count is PD[i-1].
        # The second and third subarrays start from A[i] (0-based).
        # The maximum sum of distincts for these two parts starting at index i is MaxSuffixSplit[i].
        max_total_sum = max(max_total_sum, PD[i - 1] + MaxSuffixSplit[i])

    print(max_total_sum)
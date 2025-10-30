import sys

# Use fast I/O
input = sys.stdin.readline

def solve():
    N = int(input())
    P = list(map(int, input().split()))

    # Check for 0 operations
    is_sorted = True
    for i in range(N):
        if P[i] != i + 1:
            is_sorted = False
            break
    
    if is_sorted:
        print(0)
        return

    # Check for 1 operation
    # A single operation with index k (1-indexed) sorts the array iff
    # 1. P_k = k
    # 2. The set of values {P_1, ..., P_{k-1}} is {1, ..., k-1}.
    # 3. The set of values {P_{k+1}, ..., P_N} is {k+1, ..., N}.
    # Since P is a permutation, conditions 1 and 2 imply 3, and 1 and 3 imply 2.
    # So, 1 operation is sufficient iff there exists k such that P_k = k AND {P_1, ..., P_{k-1}} = {1, ..., k-1}.
    # The set condition {P_1, ..., P_{k-1}} = {1, ..., k-1} is equivalent to
    # max(P_1, ..., P_{k-1}) <= k-1.
    # Since P_i are integers >= 1, max(P_1, ..., P_{k-1}) <= k-1 is equivalent to max(P_1, ..., P_{k-1}) < k.
    # For k=1, the prefix P_1...P_0 is empty. The maximum of an empty set is considered 0 for this context (as all values are >= 1).
    # Condition for k=1: P_1=1 AND max({}) < 1, i.e., P_1=1 AND 0 < 1. This is P_1=1.
    # For k > 1: P_k=k AND max(P_1, ..., P_{k-1}) < k.
    
    one_op_possible = False
    # current_max_prefix stores max(P_1, ..., P_{k-1}) (1-indexed values) when considering k
    # Initial value 0 is for k=1 (empty prefix max)
    current_max_prefix = 0 
    
    for i in range(N): # i is 0-indexed, corresponds to P[i]
        k = i + 1 # k is 1-indexed position
        
        if P[i] == k:
            # P_k = k. Check the prefix condition: max(P_1, ..., P_{k-1}) < k
            # current_max_prefix holds max(P_1, ..., P_{k-1}) before including P[i]
            if current_max_prefix < k:
                one_op_possible = True
                # Found a k that allows sorting in 1 op, no need to check further
                break 
        
        # Update current_max_prefix for the next iteration (for k+1)
        # current_max_prefix will store max(P_1, ..., P_k)
        current_max_prefix = max(current_max_prefix, P[i])

    if one_op_possible:
        print(1)
    else:
        # If not sorted and not sortable in 1 operation, it takes 2 operations.
        # The problem guarantees it's possible in finite steps.
        # With N >= 3, it can be shown that 2 operations are always sufficient if 0 or 1 are not.
        print(2)

T = int(input())
for _ in range(T):
    solve()
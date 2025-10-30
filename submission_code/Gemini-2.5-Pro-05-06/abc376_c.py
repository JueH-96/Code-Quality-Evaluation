import bisect

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    B_list = list(map(int, input().split())) 

    A.sort()
    B_list.sort() # B_list has N-1 elements

    # _ok_prefix[k_val] is true if A[i] <= B_list[i] for all 0 <= i < k_val
    # k_val is the insertion index of the new box x into B_list.
    # k_val can range from 0 to N-1.
    # So _ok_prefix is of size N.
    # _ok_prefix[0] signifies an empty condition (i < 0), thus true.
    # _ok_prefix[k_val] means A[0]...A[k_val-1] must fit B_list[0]...B_list[k_val-1].
    _ok_prefix = [False] * N 
    _ok_prefix[0] = True # Base case for k_val = 0
    for k_val in range(1, N): # k_val from 1 up to N-1
        # To check _ok_prefix[k_val], we need condition A[k_val-1] <= B_list[k_val-1]
        # Max k_val-1 is N-2, which is a valid index for B_list (length N-1).
        if _ok_prefix[k_val-1] and (A[k_val-1] <= B_list[k_val-1]):
            _ok_prefix[k_val] = True
        else:
            # If previous prefix was false, or current condition fails, this prefix is false.
            # All subsequent _ok_prefix entries relying on this would also effectively be false.
            _ok_prefix[k_val] = False 

    # _ok_suffix[k_val] is true if A[i+1] <= B_list[i] for all k_val <= i <= N-2
    # k_val can range from 0 to N-1.
    # So _ok_suffix is of size N.
    # _ok_suffix[N-1] means k_val=N-1. Loop N-1 <= i <= N-2 is empty, so true.
    # _ok_suffix[k_val] means A[k_val+1]...A[N-1] must fit B_list[k_val]...B_list[N-2].
    _ok_suffix = [False] * N
    _ok_suffix[N-1] = True # Base case for k_val = N-1
    for k_val in range(N-2, -1, -1): # k_val from N-2 down to 0
        # To check _ok_suffix[k_val], we need condition A[k_val+1] <= B_list[k_val]
        # Max k_val is N-2. A[k_val+1] -> A[N-1]. B_list[k_val] -> B_list[N-2]. All valid indices.
        if _ok_suffix[k_val+1] and (A[k_val+1] <= B_list[k_val]):
            _ok_suffix[k_val] = True
        else:
            _ok_suffix[k_val] = False
            
    min_valid_x = float('inf')
    
    low = 1
    high = 2 * (10**9) + 7 # Sufficiently large upper bound for x

    solution_found = False
    
    while low <= high:
        test_x = low + (high - low) // 2
        
        # k_idx is the insertion point of test_x into B_list.
        # B_list[0...k_idx-1] < test_x (if using standard bisect_left def)
        # B_list[k_idx...N-2] >= test_x
        # k_idx can range from 0 to N-1 (i.e., len(B_list)).
        k_idx = bisect.bisect_left(B_list, test_x)

        # After inserting test_x, the new box is C[k_idx]. Toy A[k_idx] must fit.
        # A[0...k_idx-1] map to B_list[0...k_idx-1]. Condition is _ok_prefix[k_idx].
        # A[k_idx+1...N-1] map to B_list[k_idx...N-2]. Condition is _ok_suffix[k_idx].
        
        current_x_can_work = True
        if not _ok_prefix[k_idx]:
            current_x_can_work = False
        
        # A[k_idx] must exist. Max k_idx is N-1. A has N elements (indices 0..N-1). So A[k_idx] is valid.
        if current_x_can_work and A[k_idx] > test_x:
            current_x_can_work = False
        
        if current_x_can_work and not _ok_suffix[k_idx]:
            current_x_can_work = False
        
        if current_x_can_work:
            min_valid_x = test_x
            solution_found = True
            high = test_x - 1 # Try to find an even smaller x
        else:
            low = test_x + 1 # test_x is too small or invalid positionally
            
    if solution_found:
        print(min_valid_x)
    else:
        print("-1")

solve()
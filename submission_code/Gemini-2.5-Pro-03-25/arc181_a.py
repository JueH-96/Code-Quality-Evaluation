# YOUR CODE HERE
import sys

# Function to solve a single test case
def solve():
    # Read the size of the permutation
    N = int(sys.stdin.readline())
    # Read the permutation P as a list of integers
    P = list(map(int, sys.stdin.readline().split()))
    
    # Adjust P to be 0-indexed for convenience in implementation.
    # The permutation P becomes a permutation of 0, 1, ..., N-1.
    # The target sorted state is (0, 1, ..., N-1).
    P = [x - 1 for x in P] 
    
    # Check if the permutation is already sorted. If so, 0 operations are required.
    is_sorted = True
    for i in range(N):
        if P[i] != i:
            is_sorted = False
            break
    
    if is_sorted:
        print(0)
        return

    # Case 1: Check if 1 operation suffices using k=1 or k=N as the pivot index.
    # If P_1 = 1 (which is P[0] == 0 in 0-based index), we can choose k=1.
    # The operation sorts P[2..N], resulting in (1, 2, ..., N).
    # If P_N = N (which is P[N-1] == N-1 in 0-based index), we can choose k=N.
    # The operation sorts P[1..N-1], resulting in (1, 2, ..., N).
    # So, if either condition P[0] == 0 or P[N-1] == N-1 holds, the answer is 1.
    if P[0] == 0 or P[N-1] == N-1:
        print(1)
        return
    
    # Case 2: Check if 1 operation suffices using an internal k (1 < k < N) as the pivot index.
    # The condition for 1 operation with pivot k is that after the operation, the array becomes sorted.
    # This happens if and only if:
    #   1. The element at position k remains correct: P_k = k.
    #   2. The elements P_1, ..., P_{k-1} are exactly the set {1, ..., k-1}.
    #   3. The elements P_{k+1}, ..., P_N} are exactly the set {k+1, ..., N}.
    # Condition 2 implies max(P_1, ..., P_{k-1}) = k-1.
    # If conditions 1 and 2 hold, condition 3 holds automatically because P is a permutation.
    # So the simplified check is: P_k = k AND max(P_1, ..., P_{k-1}) = k-1.
    
    # In 0-based indexing, let idx = k-1. The condition becomes:
    # P[idx] = idx AND max(P[0], ..., P[idx-1]) = idx-1.
    # This check needs to be performed for k from 2 to N-1, which means idx from 1 to N-2.

    # We need prefix maximums to efficiently check the max condition.
    # prefix_max_arr[i] will store max(P[0]..P[i])
    prefix_max_arr = [-1] * N  
    current_max = -1
    for i in range(N):
        current_max = max(current_max, P[i])
        prefix_max_arr[i] = current_max
    
    found_k_for_1_op = False
    # Iterate through possible internal pivot indices (0-based idx from 1 to N-2)
    # This corresponds to k from 2 to N-1 in 1-based indexing.
    for idx in range(1, N - 1): 
        # Check the condition P[idx] == idx and max(P[0]..P[idx-1]) == idx-1
        # The maximum value max(P[0]..P[idx-1]) is stored in prefix_max_arr[idx-1]
        if P[idx] == idx and prefix_max_arr[idx-1] == idx - 1:
            found_k_for_1_op = True
            break # Found a k that works, no need to check further
            
    if found_k_for_1_op:
        print(1)
        return
        
    # Case 3: Check for the specific case requiring 3 operations.
    # Based on analysis, this potentially occurs only when P_1 = N AND P_N = 1.
    # In 0-based index: P[0] == N-1 AND P[N-1] == 0.
    # It can be shown that this configuration always requires 3 operations if N >= 3.
    if P[0] == N-1 and P[N-1] == 0:
        print(3)
        return

    # Case 4: If none of the above conditions for 0, 1, or 3 operations are met, 
    # then the minimum number of operations must be 2.
    # We showed that if the permutation is not sorted, cannot be sorted in 1 step, 
    # and does not satisfy P[0]==N-1 and P[N-1]==0, then it can always be sorted in 2 steps.
    # One strategy is to make P'[0]=0 or P'[N-1]=N-1 in the first step, then sort fully in the second step.
    # This strategy works unless P[0]=N-1 and P[N-1]=0, which is handled by Case 3.
    print(2)

# Read the number of test cases
T = int(sys.stdin.readline())
# Process each test case
for _ in range(T):
    solve()
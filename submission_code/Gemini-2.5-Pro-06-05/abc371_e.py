import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        
        # Read N from stdin
        N_str = input()
        if not N_str:
            return
        N = int(N_str)

        # Read the sequence A from stdin
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle potential empty input or parsing errors
        return

    # last_pos[v] will store the 1-based index of the last occurrence of value v.
    # Values are 1 to N, so an array of size N+1 is sufficient.
    # Initialized to 0, which serves as the "previous index" for the first occurrence.
    last_pos = [0] * (N + 1)

    total_sum = 0

    # Iterate through the array A with a 1-based index k.
    for k in range(1, N + 1):
        # Current value is A[k-1] since A is 0-indexed.
        val = A[k - 1]
        
        # p_k is the 1-based index of the previous occurrence of val.
        p_k = last_pos[val]
        
        # The contribution of the element at index k is the number of subarrays
        # (i, j) where A_k is the first occurrence of its value.
        # The number of choices for the start index i is (k - p_k).
        # The number of choices for the end index j is (N - k + 1).
        contribution = (k - p_k) * (N - k + 1)
        
        # Add this contribution to the total sum.
        total_sum += contribution
        
        # Update the last seen position for val to the current 1-based index k.
        last_pos[val] = k

    print(total_sum)

solve()
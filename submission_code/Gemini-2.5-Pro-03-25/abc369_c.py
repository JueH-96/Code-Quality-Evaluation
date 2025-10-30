import sys

def solve():
    # Read the number of integers N
    N = int(sys.stdin.readline())
    
    # Handle the edge case N=1. 
    # A sequence of length 1 is always an arithmetic progression.
    # The only pair (l,r) is (1,1), so the answer is 1.
    if N == 1:
        print(1)
        # Read the single element A_1 to consume the input line, although its value is not needed.
        sys.stdin.readline() 
        return

    # Read the sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # If N=2, the pairs are (1,1), (2,2), (1,2). 
    # Sequences are (A1), (A2), (A1, A2). All lengths 1 or 2 are APs.
    # Total count is 3. The code below correctly calculates this.

    # Compute the differences between adjacent elements.
    # The difference array `diffs` will have length N-1.
    # diffs[i] = A[i+1] - A[i] (using 0-based indexing for A)
    diffs = []
    for i in range(N - 1):
        diffs.append(A[i+1] - A[i])

    # Initialize the total count of AP subsequences.
    # Start by counting all subsequences of length 1 and 2.
    # All subsequences of length 1 are APs. There are N such pairs (i, i).
    # All subsequences of length 2 are APs. There are N-1 such pairs (i, i+1).
    # Base count = N (length 1) + (N-1) (length 2).
    total_count = N + (N - 1) 

    # Now, count APs of length >= 3.
    # A subsequence A[l..r] (0-based index) is an AP of length >= 3 if and only if 
    # the differences A[i+1]-A[i] are constant for l <= i < r.
    # This means the corresponding segment in the `diffs` array, diffs[l..r-1], must consist of identical values.
    
    # We iterate through the `diffs` array to find maximal contiguous blocks of equal values.
    # For each block of length k, it corresponds to an AP segment in A of length k+1.
    # Any contiguous subsegment of this AP of length >= 3 is also an AP.
    # The number of such subsegments contributes to the total count.
    
    # This variable will accumulate the count of APs of length >= 3.
    count_len_ge_3 = 0
    
    i = 0
    # Iterate through the `diffs` array. Its length is N-1.
    while i < N - 1:
        j = i
        # Extend `j` to find the end of the current block of equal differences starting at index `i`.
        # Check `j+1` is within bounds of `diffs` array (indices 0 to N-2).
        while j + 1 < N - 1 and diffs[j+1] == diffs[i]:
            j += 1
        
        # The block of equal differences is diffs[i...j].
        # The length of this block is k = j - i + 1.
        k = j - i + 1
        
        # A block of k identical differences corresponds to an AP segment in A: A[i], A[i+1], ..., A[j+1].
        # This segment has length M = k+1.
        # We need to count the number of contiguous subsegments of A[i...j+1] that have length >= 3.
        # The number of such subsegments is Sum_{p=3}^{M} (M - p + 1).
        # This sum evaluates to (M-2)(M-1)/2.
        # Substituting M = k+1, the count becomes (k+1-2)(k+1-1)/2 = (k-1)k/2.
        
        # Add this count to `count_len_ge_3`. Uses integer division `//`.
        # Note that if k=1 (block length 1), this adds 1 * (1-1) // 2 = 0, which is correct.
        # A block of length 1 means only A[i], A[i+1], A[i+2] form an AP (length 3). 
        # The formula k * (k-1) // 2 gives 1 for k=2, corresponding to the AP of length 3.
        # Ah, careful re-read: the sum formula is correct, (k-1)k/2 represents pairs with length >= 3.
        count_len_ge_3 += k * (k - 1) // 2
        
        # Move the starting index `i` to the position right after the current block.
        i = j + 1

    # The final answer is the sum of counts for APs of lengths 1, 2, and >= 3.
    print(total_count + count_len_ge_3)

# Execute the solve function
solve()
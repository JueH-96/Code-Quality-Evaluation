import sys

# sys.setrecursionlimit(3000) # Not needed with iterative approach

def combinations(n, k):
    """Calculates the binomial coefficient C(n, k) = n! / (k! * (n-k)!),
       using iterative multiplication and division suitable for large numbers."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # Optimization: C(n, k) = C(n, n-k)
    if k > n // 2:
        k = n - k
    
    # Calculate iteratively C(n, k) = prod(n-i+1)/prod(i) for i=1 to k
    # Using integer division throughout avoids floating point issues
    # and leverages Python's arbitrary precision integers.
    res = 1
    for i in range(1, k + 1):
        # The value `res * (n - i + 1)` is always divisible by `i`
        # because `res` is C(n, i-1) and C(n, i) = C(n, i-1) * (n-i+1) / i
        # C(n, i) is an integer, so the product is divisible by i.
        res = (res * (n - i + 1)) // i
    return res

def multinomial(counts):
    """Calculates the multinomial coefficient (sum(counts))! / (counts[0]! * counts[1]! * ...).
       This is computed as a product of binomial coefficients."""
    total_len = sum(counts)
    res = 1
    current_len = total_len
    for count in counts:
        if count < 0:
            # This indicates an error in logic if it ever happens
            return 0
        if count == 0:
            continue
        
        # The multinomial coefficient C(L', c1, ..., cN) can be computed as:
        # C(L', c1) * C(L'-c1, c2) * ... * C(L' - sum(c_i for i<k), ck) * ... * C(cN, cN)
        # where L' = sum(counts)
        res = res * combinations(current_len, count)
        current_len -= count
        
    return res

def solve():
    """
    Finds the floor((S+1)/2)-th lexicographically smallest good integer sequence.
    A good sequence has length NK and contains each integer from 1 to N exactly K times.
    S is the total number of such sequences.
    """
    line = sys.stdin.readline().split()
    N = int(line[0])
    K = int(line[1])

    L = N * K
    # Use a list to store the remaining counts for each number (1 to N).
    # counts[i] will store the remaining count for number (i+1).
    counts = [K] * N

    # Calculate the total number of good sequences (S).
    # This is the multinomial coefficient for the initial counts [K, K, ..., K].
    S = multinomial(counts)

    # Calculate the target rank R (1-based) for the sequence we want to find.
    R = (S + 1) // 2

    sequence = [] # List to store the resulting sequence
    C_rem = S           # Total number of permutations of the remaining elements.
                        # Initially, this is the total number of good sequences S.
    L_rem = L           # Remaining length of the sequence to construct.
                        # Initially, this is the total length L = NK.

    # We determine the sequence element by element, from left to right.
    # The loop runs L times, once for each position in the sequence.
    for step in range(L):
        
        # Find the element for the current position (step).
        # We try possible elements (1 to N) in increasing order to ensure lexicographical search.
        found_element = -1
        
        # Iterate through possible elements (1 to N).
        # i goes from 0 to N-1, representing elements 1 to N.
        for i in range(N):
            # If element (i+1) can still be used (its count is > 0).
            if counts[i] > 0:
                # Calculate the number of sequences that would start with element (i+1)
                # at the current position. This is the number of permutations of the
                # remaining L_rem-1 elements after picking element (i+1).
                # The counts for the remaining elements are the current counts,
                # except counts[i] is one less.
                # The number of permutations of L' items with counts c1, ..., cN is C(L', c1, ..., cN).
                # If we pick element i (1-based), the remaining counts are counts_new
                # where counts_new[i-1] = counts[i-1] - 1. The remaining length is L_rem - 1.
                # The number of permutations of these is C(L_rem - 1, counts_new).
                # This count is related to the current total C_rem by the formula:
                # C(L_rem - 1, counts_new) = C_rem * counts[i] / L_rem (using integer division)
                
                Count_if_i = (C_rem * counts[i]) // L_rem
                
                # Check if the target sequence falls within the block of sequences
                # that start with element (i+1) at the current position.
                # If the rank R is less than or equal to the number of sequences
                # starting with element (i+1), then the element for the current position is (i+1).
                if R <= Count_if_i:
                    # Element (i+1) is the current element in the target sequence.
                    sequence.append(i + 1)
                    counts[i] -= 1     # Decrease the count for the chosen element.
                    L_rem -= 1         # Decrease the remaining length.
                    C_rem = Count_if_i # The total number of permutations for the *next* step
                                       # is the number of sequences we found starting with this element.
                    break              # Found the element for this position, move to determine the next element.
                else:
                    # Element (i+1) is not the current element.
                    # The target sequence is in a later lexicographical block.
                    # Subtract the number of sequences in the block we just skipped (those starting with element (i+1)).
                    R -= Count_if_i
                    # C_rem remains the same when considering different elements at the current position.

    # Print the resulting sequence elements separated by spaces.
    # Use map(str, sequence) to convert integers to strings before joining/printing.
    print(*sequence)

# Read input and run the solver
solve()
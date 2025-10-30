# YOUR CODE HERE
import sys

def is_mountain(k):
    """
    Determines if the k-th crease is a mountain fold.
    The k-th crease from the left (1-indexed).
    k must be a positive integer.
    
    A crease is a mountain fold if and only if the odd part of k
    (i.e., k / 2^v_2(k)) is congruent to 3 modulo 4.
    
    `k & -k` isolates the lowest set bit of k, which is 2^v_2(k).
    """
    if k <= 0:
        return False
    
    odd_part = k // (k & -k)
    return odd_part % 4 == 3

def solve():
    """
    Main function to solve the problem.
    """
    try:
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # The function f(i) is the number of mountain folds among creases i+A_1, ..., i+A_N.
    # A direct search for i is impossible due to the large range.
    # The maximum of f(i) likely occurs when one of the arguments i + A_j aligns
    # with a canonical number that produces a mountain fold.
    # The simplest such numbers are of the form 3 * 2^p.
    # We test candidate i values of the form i = 3 * 2^p - A_j.

    max_f = 0
    if N > 0:
        max_f = 1 # At least one crease can always be made a mountain fold.

    # Iterate through each A_j as a potential anchor point
    for j in range(N):
        # A_k <= 10^18 < 2^60. Differences A_j - A_k are of a similar magnitude.
        # p up to 63 covers interactions with the lower 64 bits.
        for p in range(64):
            # Candidate i, chosen to make i + A_j = 3 * 2^p
            # 1 << p computes 2^p
            candidate_i = (3 * (1 << p)) - A[j]
            
            # The problem states i >= 1.
            if candidate_i <= 0:
                continue

            current_f = 0
            for k in range(N):
                val = candidate_i + A[k]
                if is_mountain(val):
                    current_f += 1
            
            if current_f > max_f:
                max_f = current_f

    print(max_f)

solve()
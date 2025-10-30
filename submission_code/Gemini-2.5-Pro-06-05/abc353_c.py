# YOUR CODE HERE
import sys

# Set up fast I/O
input = sys.stdin.readline

def solve():
    """
    Reads input, solves the problem, and prints the result.
    """
    # Read problem inputs
    try:
        N = int(input())
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        return

    # The modulus value as defined in the problem
    MOD = 10**8

    # The problem asks for S = sum_{i<j} (A_i + A_j) % MOD.
    # This can be rewritten as:
    # S = sum_{i<j} (A_i + A_j) - MOD * (number of pairs (i, j) with i<j where A_i + A_j >= MOD)

    # Part 1: Calculate sum_{i<j} (A_i + A_j)
    # Each element A_k is part of (N-1) pairs, so it's added (N-1) times.
    # This sum equals (N-1) * sum(A).
    total_sum_unmodded = (N - 1) * sum(A)

    # Part 2: Calculate C, the number of pairs (i, j) with i < j where A_i + A_j >= MOD.
    # To do this efficiently, we first sort A.
    A.sort()

    # We use a two-pointer technique on the sorted array.
    C = 0
    left = 0
    right = N - 1
    while left < right:
        if A[left] + A[right] >= MOD:
            # The pair (A[left], A[right]) satisfies the condition.
            # Since A is sorted, any A[k] with left <= k < right will also satisfy
            # A[k] + A[right] >= MOD.
            # These are pairs (A[left], A[right]), (A[left+1], A[right]), ..., (A[right-1], A[right]).
            # There are (right - 1) - left + 1 = right - left such pairs.
            C += (right - left)
            
            # We have now counted all pairs involving A[right]. Move to the next element.
            right -= 1
        else:
            # A[left] + A[right] < MOD. The sum is too small.
            # To increase the sum, we must pick a larger element than A[left].
            left += 1
            
    # The final answer is the unmodded sum minus the total subtractions for the modulo.
    result = total_sum_unmodded - C * MOD

    # Print the final result to standard output.
    print(result)

solve()
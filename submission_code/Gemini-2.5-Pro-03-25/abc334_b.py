# YOUR CODE HERE
import sys

def solve():
    # Read input values A, M, L, R from standard input
    # A: coordinate of one of the trees
    # M: interval between adjacent trees
    # L: coordinate of Takahashi (left endpoint of the interval)
    # R: coordinate of Aoki (right endpoint of the interval, L <= R)
    A, M, L, R = map(int, sys.stdin.readline().split())

    # The coordinates of the Christmas trees are given by A + k*M for any integer k.
    # We need to find the number of integers k such that the coordinate X = A + k*M satisfies L <= X <= R.
    
    # The condition L <= A + k*M <= R can be rewritten by subtracting A from all parts:
    # L - A <= k*M <= R - A
    
    # Since M is guaranteed to be positive (M >= 1), we can divide by M without changing the direction of the inequalities:
    # (L - A) / M <= k <= (R - A) / M

    # We are looking for the number of integers k that fall within the closed interval [(L - A) / M, (R - A) / M].
    
    # Let's use a method based on finding the count of points up to a certain coordinate.
    # While the total number of trees is infinite, we can find the count within a finite interval [L, R] 
    # by considering the difference between counts up to R and counts up to L-1.
    
    # Let k_max(X) be the largest integer k such that A + k*M <= X.
    # This inequality can be rearranged:
    # k*M <= X - A
    # k <= (X - A) / M  (since M > 0)
    # The largest integer k satisfying this is floor((X - A) / M).
    # So, k_max(X) = floor((X - A) / M).

    # The set of integers k corresponding to trees with coordinates in [L, R] are those k such that:
    # A + k*M <= R  AND  A + k*M >= L.
    # The first inequality means k <= k_max(R).
    # The second inequality A + k*M >= L is equivalent to saying that A + k*M is NOT <= L - 1.
    # This means k cannot be less than or equal to k_max(L-1). In other words, k must be strictly greater than k_max(L-1).
    # So, we are looking for integers k such that k_max(L-1) < k <= k_max(R).
    
    # The integers k in the range (k_max(L-1), k_max(R)] are:
    # k_max(L-1) + 1, k_max(L-1) + 2, ..., k_max(R).
    # The number of such integers is k_max(R) - (k_max(L-1) + 1) + 1 = k_max(R) - k_max(L-1).

    # Python's integer division operator `//` performs floor division. 
    # `a // b` computes floor(a / b). This works correctly for both positive and negative `a`, provided `b > 0`.
    # In this problem, M >= 1, so M > 0.

    # Calculate k_max(R) = floor((R - A) / M)
    # This can be computed using Python's integer division:
    k_max_R = (R - A) // M
    
    # Calculate k_max(L-1) = floor((L - 1 - A) / M)
    # This can also be computed using Python's integer division:
    # Note: We use L-1 because we want to subtract the count of trees strictly to the left of L.
    k_max_L_minus_1 = (L - 1 - A) // M
    
    # The total count of trees in the interval [L, R] is the difference
    count = k_max_R - k_max_L_minus_1
    
    # Print the final count to standard output
    print(count)

# Execute the solve function to run the logic
solve()
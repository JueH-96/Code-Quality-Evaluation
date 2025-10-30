import sys
from bisect import bisect_left, bisect_right

def main():
    """
    Reads input, solves the problem, and prints the result.
    """
    # Read problem inputs from standard input for efficiency
    try:
        N, M = map(int, sys.stdin.readline().split())
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handles cases like empty input at the end of a file in some test systems.
        return

    # Sort the lists to allow for efficient counting using binary search (bisect module).
    A.sort()
    B.sort()

    def check(price):
        """
        Check if a given price satisfies the problem's condition.
        Condition: (number of sellers willing to sell at 'price') >= 
                   (number of buyers willing to buy at 'price')
        """
        # A seller `i` can sell if their minimum price A_i is <= `price`.
        # Count elements in sorted A <= price. `bisect_right` gives the count.
        seller_count = bisect_right(A, price)

        # A buyer `j` can buy if their maximum price B_j is >= `price`.
        # Count elements in sorted B >= price. This is M - (count of elements < price).
        # `bisect_left` gives the count of elements < price.
        buyer_count = M - bisect_left(B, price)

        return seller_count >= buyer_count

    # Binary search for the minimum integer X that satisfies `check(X)`.
    # The `check` function is monotonic, which makes binary search applicable.
    
    # We use a binary search pattern that maintains two bounds:
    # `ng` (not good): a price for which `check(ng)` is False.
    # `ok` (good): a price for which `check(ok)` is True.
    # The goal is to find the smallest `ok`.

    # Initialize bounds:
    # A price of 0 is not possible, and `check(0)` is False (0 sellers vs. M buyers).
    # So, 0 is a safe `ng` bound.
    ng = 0
    # A price of 10^9 + 2 will always satisfy the condition (N sellers vs. 0 buyers).
    # So, it's a safe `ok` bound.
    ok = 10**9 + 2

    # Binary search loop narrows the [ng, ok] range until they are adjacent.
    while ok - ng > 1:
        mid = (ok + ng) // 2
        if check(mid):
            # `mid` satisfies the condition, so it could be the answer.
            # We try to find a smaller valid price. `mid` becomes the new `ok` bound.
            ok = mid
        else:
            # `mid` does not satisfy the condition. The answer must be larger.
            # `mid` becomes the new `ng` bound.
            ng = mid

    # The loop terminates when `ok - ng == 1`.
    # `ok` is the smallest integer price found that satisfies the condition.
    print(ok)

if __name__ == "__main__":
    main()
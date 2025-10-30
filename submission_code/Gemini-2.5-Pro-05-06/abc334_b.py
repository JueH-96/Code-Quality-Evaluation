import math

def solve():
    A, M, L, R = map(int, input().split())

    # Tree locations are X = A + k*M for any integer k.
    # We need to find the number of trees such that L <= X <= R.
    # So, L <= A + k*M <= R.

    # Adjust relative to A:
    # L - A <= k*M <= R - A
    L_adj = L - A
    R_adj = R - A

    # Since M >= 1, M is positive. We can divide by M:
    # (L - A) / M <= k <= (R - A) / M

    # The smallest integer k (k_min) is ceil((L - A) / M).
    # The largest integer k (k_max) is floor((R - A) / M).

    # Calculate k_min using integer arithmetic for ceil(num / den) where den > 0:
    # k_min = ceil(L_adj / M)
    # This can be calculated as (L_adj + M - 1) // M using integer division.
    # This formula works for positive and negative L_adj.
    k_min = (L_adj + M - 1) // M
    
    # Calculate k_max using integer arithmetic for floor(num / den):
    # k_max = floor(R_adj / M)
    # Python's // operator performs floor division.
    k_max = R_adj // M

    # The number of integers k in the range [k_min, k_max] is k_max - k_min + 1.
    # If k_max < k_min, then there are no solutions for k, so the count is 0.
    if k_max < k_min:
        print(0)
    else:
        count = k_max - k_min + 1
        print(count)

if __name__ == '__main__':
    solve()
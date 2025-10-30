import sys

def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    # The problem asks for the minimum cost to sort the permutation P
    # into ascending order (1, 2, ..., N) using adjacent swaps.
    # A swap of P_i and P_{i+1} costs i (using 1-based indexing).
    # In 0-based indexing for a list P, swapping P[i] and P[i+1] costs i+1.

    # The target state is that the value v is at index v (1-based).
    # So, in 0-based indexing, the value v is at index v-1.

    # Consider the element currently at 0-based index i, which has the value P[i].
    # Its current 1-based index is i + 1.
    # Its target 1-based index is P[i] (the value itself).

    # The minimum cost for this problem is known to be the sum of the absolute
    # differences between the initial position (1-based index) of each element
    # and its target position (which is the value of the element itself, 1-based).
    # That is, sum |initial_index(v) - target_index(v)| over all values v.
    # For an element with value P[i] currently at 0-based index i, its initial
    # 1-based index is i+1, and its target 1-based index is P[i].
    # The contribution for this element is |(i + 1) - P[i]|.
    # The total cost is the sum of these contributions for all i from 0 to N-1.

    total_cost = 0
    for i in range(N):
        current_value = P[i]
        current_index_1based = i + 1
        target_index_1based = current_value

        total_cost += abs(current_index_1based - target_index_1based)

    print(total_cost)

solve()
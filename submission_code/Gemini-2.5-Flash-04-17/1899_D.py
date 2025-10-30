import sys
from collections import Counter

# Use sys.stdin.readline for faster input
input = sys.stdin.readline

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    # We need to count pairs of indices (i, j) with i < j such that b_i^b_j = b_j^b_i.
    # Substituting b_k = 2^a_k, the condition becomes (2^a_i)^(2^a_j) = (2^a_j)^(2^a_i),
    # which simplifies to 2^(a_i * 2^a_j) = 2^(a_j * 2^a_i).
    # Since the base is 2 (positive), the exponents must be equal:
    # a_i * 2^a_j = a_j * 2^a_i.
    # Given a_k >= 1, this equality holds if and only if a_i = a_j OR {a_i, a_j} = {1, 2}.
    # This can be shown by analyzing the function f(x) = 2^x / x for x >= 1.
    # f(1) = 2, f(2) = 2. For integers x >= 3, f(x) is strictly increasing, and f(x) > 2.
    # Thus, f(a_i) = f(a_j) if and only if a_i = a_j or {a_i, a_j} = {1, 2}.

    # Count pairs (i, j) with i < j such that a[i] == a[j].
    # If a value 'x' appears 'c' times, there are c * (c - 1) // 2 pairs of indices (i, j) with i < j such that a[i] = a[j] = x.
    counts = Counter(a)
    count_equal = 0
    for count in counts.values():
        if count >= 2:
            count_equal += count * (count - 1) // 2

    # Count pairs (i, j) with i < j such that {a[i], a[j]} == {1, 2}.
    # This means (a[i]=1 and a[j]=2) or (a[i]=2 and a[j]=1).
    # We can iterate through the array and for each element a[j]:
    # If a[j] is 2, it forms a valid pair (i, j) with any previous element a[i] where i < j and a[i] = 1.
    # If a[j] is 1, it forms a valid pair (i, j) with any previous element a[i] where i < j and a[i] = 2.
    count_one_two = 0
    count_of_ones = 0
    count_of_twos = 0
    for x in a:
        if x == 2:
            # Current element is 2 at index j (implicit).
            # It forms a pair (i, j) with a[j]=2 with all previous elements a[i] where i < j and a[i]=1.
            # The number of such previous elements is count_of_ones.
            count_one_two += count_of_ones
            count_of_twos += 1
        elif x == 1:
            # Current element is 1 at index j (implicit).
            # It forms a pair (i, j) with a[j]=1 with all previous elements a[i] where i < j and a[i] == 2.
            # The number of such previous elements is count_of_twos.
            count_one_two += count_of_twos
            count_of_ones += 1
        # Elements other than 1 or 2 do not form the {1, 2} pair with the current element.
        # We still need to keep track of the counts of 1s and 2s seen so far for future elements.
        # The current logic correctly updates counts_of_ones/twos only when x is 1 or 2.
        # This is sufficient because only 1s and 2s contribute to the running counts needed for count_one_two.

    # The total count is the sum of count_equal and count_one_two.
    # The two conditions (a[i] == a[j]) and ({a[i], a[j]} == {1, 2}) are mutually exclusive for a pair of values (a[i], a[j]) where a[i], a[j] >= 1.
    # If a[i] == a[j], then {a[i], a[j]} = {a[i]}. This cannot be {1, 2} as {1, 2} contains two distinct elements.
    # If {a[i], a[j]} == {1, 2}, then a[i] != a[j].
    # Thus, the set of index pairs (i, j) with i < j satisfying a[i] == a[j] is disjoint from the set of index pairs (i, j) with i < j satisfying {a[i], a[j]} == {1, 2}.
    # We can simply add the counts.
    print(count_equal + count_one_two)

t = int(input())
for _ in range(t):
    solve()
import sys
from collections import Counter, defaultdict

def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read the array A
    A = list(map(int, sys.stdin.readline().split()))

    # This problem asks for the number of triples (i, j, k) such that
    # 1 <= i < j < k <= N, A[i-1] == A[k-1], and A[i-1] != A[j-1].
    # Using 0-based indexing for the array A, the conditions are:
    # 0 <= i' < j' < k' <= N-1, A[i'] == A[k'], and A[i'] != A[j'],
    # where i' = i-1, j' = j-1, k' = k-1.

    # We can iterate through the middle index j' (0-based array index).
    # For a fixed j', let v_j = A[j']. We need to count pairs (i', k') such that
    # 0 <= i' < j' < k' <= N-1, A[i'] == A[k'], and A[i'] != v_j.

    # For a fixed j' and a fixed value v such that v != v_j, the number of pairs (i', k') is:
    # (number of times v appears in A[0...j'-1]) * (number of times v appears in A[j'+1...N-1]).

    # Let counts_total[v] be the total count of value v in array A.
    # Let counts_before[v] be the count of value v in A[0...j'-1].
    # The count of value v in A[j'+1...N-1] is counts_total[v] - counts_before[v] - (1 if v == v_j else 0).
    # Since we consider v != v_j, the count after j' is counts_total[v] - counts_before[v].

    # The contribution to the total count for a fixed j' is the sum over all values v != v_j:
    # Sum_{v != v_j} counts_before[v] * (counts_total[v] - counts_before[v]).

    # We can compute this sum efficiently by iterating j' from 0 to N-1, maintaining counts_before.
    # The sum Sum_{v != v_j} counts_before[v] * (counts_total[v] - counts_before[v])
    # can be rewritten as:
    # (Sum_{v} counts_before[v] * (counts_total[v] - counts_before[v])) - (counts_before[v_j] * (counts_total[v_j] - counts_before[v_j]))
    # The first term is Sum_{v} (counts_before[v] * counts_total[v] - counts_before[v]^2).
    # Let sum_prod_v_total = Sum_{v} counts_before[v] * counts_total[v]
    # Let sum_sq_before = Sum_{v} counts_before[v]^2
    # We can update these two sums iteratively as we increment counts_before.

    counts_total = Counter(A)
    total_count = 0 # Python handles large integers

    # counts_before will store counts of elements in A[0...j'-1]
    counts_before = Counter()

    # sum_prod_v_total = Sum_{v} counts_before[v] * counts_total[v]
    sum_prod_v_total = 0

    # sum_sq_before = Sum_{v} counts_before[v]^2
    sum_sq_before = 0

    # Iterate through the array elements by their 0-based index j'
    # This index j' corresponds to the middle element A[j'] in the triple (i', j', k').
    # The indices i' are < j', and k' are > j'.
    for j_prime in range(N):
        v_j = A[j_prime]

        # Calculate the contribution for this j'.
        # The sum is Sum_{v != v_j} counts_before[v] * (counts_total[v] - counts_before[v]).
        # This sum equals (Sum_{v} counts_before[v] * (counts_total[v] - counts_before[v])) - (counts_before[v_j] * (counts_total[v_j] - counts_before[v_j])).
        # Note that counts_before[v_j] here is the count of v_j in A[0...j'-1], which is the current state of counts_before.

        count_v_j_before = counts_before[v_j]
        count_v_j_total = counts_total[v_j]

        # The term (sum_prod_v_total - sum_sq_before) is equal to Sum_{v} counts_before[v] * (counts_total[v] - counts_before[v]),
        # using counts_before before processing A[j_prime].
        # The term (count_v_j_before * (count_v_j_total - count_v_j_before)) is the part of the sum where v == v_j.
        # Subtracting this term gives the sum over v != v_j.
        current_contrib = (sum_prod_v_total - sum_sq_before) - (count_v_j_before * (count_v_j_total - count_v_j_before))
        total_count += current_contrib

        # Now, update sum_prod_v_total and sum_sq_before for the next iteration (j' + 1).
        # A[j'] is moved from "after" to "before". The count of v_j in counts_before increments by 1.
        # For any v != v_j, the count in counts_before remains unchanged.

        # Update sum_prod_v_total:
        # The old value was Sum_{v} counts_before[v] * counts_total[v].
        # The new value will be Sum_{v != v_j} counts_before[v] * counts_total[v] + (counts_before[v_j] + 1) * counts_total[v_j].
        # Change = [(counts_before[v_j] + 1) * counts_total[v_j]] - [counts_before[v_j] * counts_total[v_j]] = counts_total[v_j].
        sum_prod_v_total += count_v_j_total

        # Update sum_sq_before:
        # The old value was Sum_{v} counts_before[v]^2.
        # The new value will be Sum_{v != v_j} counts_before[v]^2 + (counts_before[v_j] + 1)^2.
        # Change = (counts_before[v_j] + 1)^2 - counts_before[v_j]^2 = counts_before[v_j]^2 + 2 * counts_before[v_j] + 1 - counts_before[v_j]^2 = 2 * counts_before[v_j] + 1.
        sum_sq_before += 2 * count_v_j_before + 1

        # Finally, update counts_before for the next iteration by including A[j'].
        counts_before[v_j] += 1

    print(total_count)

solve()
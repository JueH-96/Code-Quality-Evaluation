# YOUR CODE HERE
from collections import defaultdict
import sys

# Read input faster
input = sys.stdin.readline

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Step 1: Precompute positions
    # pos[val] stores a list of 0-indexed positions where val appears in A
    # The list will have exactly two elements, sorted.
    pos = defaultdict(list)
    for i in range(2 * N):
        pos[A[i]].append(i)

    # Step 2: Identify numbers that are initially adjacent
    # These numbers violate condition 1 or 2 if they are part of the pair (a, b)
    adjacent_nums = set()
    # Iterate through numbers 1 to N
    for val in range(1, N + 1):
        # Check if the two occurrences are adjacent
        if pos[val][1] - pos[val][0] == 1:
            adjacent_nums.add(val)

    # Step 3: Group indices k by canonical pair {A[k], A[k+1]}
    # We are looking for pairs (a, b) where the original positions {p1a, p2a, p1b, p2b}
    # form two adjacent pairs {k, k+1, l, l+1} with k+1 < l.
    # This structure requires the multiset {A[k], A[k+1]} to be equal to {A[l], A[l+1]} and be {a, b}.
    # We collect all indices k where {A[k], A[k+1]} is a specific multiset {v1, v2} with v1 != v2.
    k_indices = defaultdict(list)
    # Iterate up to 2*N - 1 to consider all adjacent pairs (A[k], A[k+1])
    for k in range(2 * N - 1):
        v1, v2 = A[k], A[k+1]
        # We only care about adjacent positions holding distinct values.
        # If v1 == v2, these are already adjacent and cannot form {a, b} unless a=b,
        # which is not allowed for the pair (a, b).
        if v1 != v2:
            # Use a canonical representation for the multiset {v1, v2}
            # Since v1 != v2, min(v1, v2) < max(v1, v2)
            canonical_pair = (min(v1, v2), max(v1, v2))
            k_indices[canonical_pair].append(k)

    # Step 4 & 5: Count valid pairs (a, b)
    count = 0
    # Iterate through each distinct canonical pair (v1, v2) found in adjacent positions
    # These (v1, v2) are the potential (a, b) pairs.
    # The problem asks for pairs (a, b) with a < b. The canonical pair (v1, v2)
    # with v1 < v2 ensures this condition.
    for (v1, v2), k_list in k_indices.items():
        # Check Conditions 1 and 2: v1 and v2 must not be initially adjacent
        # We computed adjacent_nums in Step 2.
        if v1 in adjacent_nums or v2 in adjacent_nums:
            continue

        # Check Condition 3: The four original positions of v1 and v2 {pos[v1][0], pos[v1][1], pos[v2][0], pos[v2][1]}
        # must form two adjacent pairs {k, k+1, l, l+1} with k+1 < l.
        # This structure is formed if and only if there exist k_p, k_q in k_list
        # (where k_list contains indices i such that {A[i], A[i+1]} = {v1, v2})
        # such that p < q and k_q > k_p + 1.
        # This condition holds if and only if the sorted list k_list has length >= 2
        # AND the indices in k_list do not form a single contiguous block.
        # A list [k_0, k_1, ..., k_{m-1}] forms a single contiguous block iff m >= 2 AND k_i = k_0 + i for all i.
        # This is equivalent to m >= 2 AND k_{m-1} = k_0 + (m-1), or k_list[-1] - k_list[0] == len(k_list) - 1.
        # So, the condition is len(k_list) >= 2 AND k_list[-1] - k_list[0] != len(k_list) - 1.

        # k_list is already sorted because we appended k in increasing order of k.
        if len(k_list) >= 2:
            # If the list has length >= 2, check if it forms a contiguous block.
            # If it does *not* form a contiguous block, there must be a gap,
            # meaning there exist k_p, k_q with k_q > k_p + 1.
            if k_list[-1] - k_list[0] != len(k_list) - 1:
                 # This pair (v1, v2) is valid.
                 count += 1

    print(count)

T = int(input())
for _ in range(T):
    solve()
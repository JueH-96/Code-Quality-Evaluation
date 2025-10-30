import sys
from collections import defaultdict

def solve():
    S = sys.stdin.readline().strip()
    n = len(S)

    # Store 0-based indices for each character
    char_indices = defaultdict(list)
    for i in range(n):
        char_indices[S[i]].append(i)

    total_count = 0

    # The problem asks for triples (i, j, k) such that 1 <= i < j < k <= |S|
    # and S[i-1]S[j-1]S[k-1] is a palindrome.
    # In 0-based indexing, this is triples (i', j', k') such that 0 <= i' < j' < k' < |S|
    # and S[i']S[j']S[k'] is a palindrome.
    # A 3-character string S[i']S[j']S[k'] is a palindrome if S[i'] == S[k'].
    # The condition on j' is simply i' < j' < k'.
    # For a fixed pair of indices (i', k') with 0 <= i' < k' < |S| and S[i'] == S[k'],
    # the number of valid j' is the number of integers between i' and k', exclusive.
    # This count is k' - i' - 1.
    # The total number of triples is the sum of (k' - i' - 1) over all pairs (i', k')
    # such that 0 <= i' < k' < |S| and S[i'] == S[k'].

    # We can group indices by character. For a character C, let its indices be
    # pos = [p_0, p_1, ..., p_{m-1}], where 0 <= p_0 < p_1 < ... < p_{m-1} < |S|.
    # We need to sum (p_b - p_a - 1) over all pairs (a, b) such that 0 <= a < b <= m-1.
    # This sum can be calculated efficiently:
    # sum_{0 <= a < b <= m-1} (p_b - p_a - 1)
    # = sum_{b=1 to m-1} sum_{a=0 to b-1} (p_b - p_a - 1)
    # For a fixed b, the inner sum is:
    # sum_{a=0 to b-1} p_b - sum_{a=0 to b-1} p_a - sum_{a=0 to b-1} 1
    # = b * p_b - (sum_{a=0 to b-1} p_a) - b

    for char, pos in char_indices.items():
        m = len(pos)
        if m < 2:
            # Need at least two occurrences of a character to form a pair (i', k') with i' < k'
            continue

        char_total = 0
        
        # We need the sum of pos[a] for a = 0 to b-1 when calculating the term for pos[b].
        # We can maintain a running prefix sum of the pos list.
        # Initialize prefix sum with the first element.
        current_prefix_sum = pos[0] # This is sum of pos[0]

        # Iterate through b from 1 to m-1. pos[b] is the current k' index.
        # The potential i' indices are pos[0], ..., pos[b-1].
        for b in range(1, m):
            # sum_{a=0 to b-1} pos[a] is the sum of elements from pos[0] up to pos[b-1].
            # This sum is exactly what current_prefix_sum holds from the previous iteration.
            sum_pos_a = current_prefix_sum

            # The contribution for the current pair (pos[a] for a in 0..b-1, pos[b])
            # is sum_{a=0 to b-1} (pos[b] - pos[a] - 1)
            # = (b * pos[b]) - (sum_{a=0 to b-1} pos[a]) - b
            term = b * pos[b] - sum_pos_a - b
            char_total += term

            # Update current_prefix_sum by adding pos[b] for the next iteration (b+1)
            current_prefix_sum += pos[b]

        total_count += char_total

    print(total_count)

solve()
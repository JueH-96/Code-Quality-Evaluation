import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    m = N // 2

    odd_indexed_vals = []
    even_indexed_vals = []
    for i in range(N):
        if (i + 1) % 2 != 0: # 1-based index is odd
            odd_indexed_vals.append(A[i])
        else: # 1-based index is even
            even_indexed_vals.append(A[i])

    odd_indexed_vals.sort()
    even_indexed_vals.sort()

    # Prefix sums for sorted values (0-indexed array)
    # prefix_sum[i] = sum of first i elements
    # prefix_sum[0] = 0
    odd_prefix_sum = [0] * (len(odd_indexed_vals) + 1)
    for i in range(len(odd_indexed_vals)):
        odd_prefix_sum[i+1] = odd_prefix_sum[i] + odd_indexed_vals[i]

    even_prefix_sum = [0] * (len(even_indexed_vals) + 1)
    for i in range(len(even_indexed_vals)):
        even_prefix_sum[i+1] = even_prefix_sum[i] + even_indexed_vals[i]

    max_score = 0

    if N % 2 == 0:
        # N = 2m, |O| = m, |E| = m
        # k is number of smallest from O with coeff -1 (0 <= k <= m)
        # (m-k) is number of largest from O with coeff +1
        # k is number of largest from E with coeff +1
        # (m-k) is number of smallest from E with coeff -1
        # Score = (S_O,large(m-k) - S_O,small(k)) + (S_E,large(k) - S_E,small(m-k))
        # S_small[j] = sum of first j elements = prefix_sum[j]
        # S_large[j] = sum of last j elements = total_sum - sum of first (total_len - j) elements
        # S_O,large(m-k) = odd_prefix_sum[m] - odd_prefix_sum[k]
        # S_O,small(k) = odd_prefix_sum[k]
        # S_E,large(k) = even_prefix_sum[m] - even_prefix_sum[m-k]
        # S_E,small(m-k) = even_prefix_sum[m-k]
        # Score(k) = (odd_prefix_sum[m] - odd_prefix_sum[k] - odd_prefix_sum[k]) + (even_prefix_sum[m] - even_prefix_sum[m-k] - even_prefix_sum[m-k])
        # Score(k) = (odd_prefix_sum[m] - 2 * odd_prefix_sum[k]) + (even_prefix_sum[m] - 2 * even_prefix_sum[m-k])
        for k in range(m + 1):
            # k represents |N_O| and |P_E| in the derivation leading to this formula
            # (m-k) represents |P_O| and |N_E|
            score = (odd_prefix_sum[m] - 2 * odd_prefix_sum[k]) + (even_prefix_sum[m] - 2 * even_prefix_sum[m-k])
            max_score = max(max_score, score)

    else: # N is odd
        # N = 2m+1, |O| = m+1, |E| = m
        # An odd-indexed element remains. Iterate through each possibility of which odd-indexed element remains.
        # The remaining element is o_j (0-indexed) from sorted odd_indexed_vals.
        # The set of removed odd indices is O_val \ {o_j}, size m.
        # The set of removed even indices is E_val, size m.
        # For fixed remaining o_j and fixed k in {0, ..., m}:
        # k is number of smallest from O \ {o_j} with coeff -1
        # m-k is number of largest from O \ {o_j} with coeff +1
        # k is number of largest from E with coeff +1
        # m-k is number of smallest from E with coeff -1
        # Score = S_{O \setminus \{o_j\}, large}(m-k) - S_{O \setminus \{o_j\}, small}(k) + S_{E,large}(k) - S_{E,small}(m-k).
        # S_{O \setminus \{o_j\}, large}(m-k) - S_{O \setminus \{o_j\}, small}(k) = Sum(O \ {o_j}) - 2 * S_{O \setminus \{o_j\}, small}(k)
        # Sum(O \ {o_j}) = odd_prefix_sum[m+1] - o_j
        # Need to minimize o_j + 2 * S_{O \setminus \{o_j\}, small}(k) for fixed k over j in {0, ..., m}.
        # Min_cost(k) = min_{j=0..m} (o_j + 2 * S_{O \setminus \{o_j\}, small}(k))
        # S_{O \setminus \{o_j\}, small}(k) = sum of smallest k from {o_0, ..., o_m} \ {o_j}

        for k in range(m + 1):
            min_odd_cost_term = float('inf') # Represents the minimum of o_j + 2 * S_{O \ {o_j}},small(k)

            # Case k = 0: sum is 0. Cost is o_j. Min is o_0.
            if k == 0:
                 min_odd_cost_term = odd_indexed_vals[0]
            
            # Case k > 0: (1 <= k <= m)
            else:
                # Case j >= k (j in [k, m]): S_{O \setminus \{o_j\}, small}(k) = odd_prefix_sum[k]. Cost = o_j + 2 * odd_prefix_sum[k].
                # Min cost over j>=k is o_k + 2 * odd_prefix_sum[k]. This is valid if k <= m.
                min_cost_j_ge_k = odd_indexed_vals[k] + 2 * odd_prefix_sum[k]
                min_odd_cost_term = min(min_odd_cost_term, min_cost_j_ge_k)

                # Case j < k (j in [0, k-1]): S_{O \setminus \{o_j\}, small}(k) = odd_prefix_sum[k+1] - o_j. Cost = 2 * odd_prefix_sum[k+1] - o_j.
                # Min cost over j<k is 2 * odd_prefix_sum[k+1] - o_{k-1}. This is valid if k >= 1 and k-1 >= 0 (i.e., k >= 1) and k+1 <= m+1 (i.e., k <= m).
                # So valid for 1 <= k <= m.
                min_cost_j_lt_k = 2 * odd_prefix_sum[k+1] - odd_indexed_vals[k-1]
                min_odd_cost_term = min(min_odd_cost_term, min_cost_j_lt_k)
            
            # Calculate score for this k and the optimal remaining odd element (which minimizes the cost term)
            # Score = (Total sum of O) - Min_cost(k) + (S_E,large(k) - S_E,small(m-k))
            total_sum_O = odd_prefix_sum[m+1]
            
            # S_E,large(k) = even_prefix_sum[m] - even_prefix_sum[m-k]
            # S_E,small(m-k) = even_prefix_sum[m-k]
            score = (total_sum_O - min_odd_cost_term) + (even_prefix_sum[m] - 2 * even_prefix_sum[m-k])
            max_score = max(max_score, score)

    print(max_score)

solve()
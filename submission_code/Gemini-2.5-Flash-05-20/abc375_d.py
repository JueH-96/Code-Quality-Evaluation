import sys

def solve():
    S = sys.stdin.readline().strip()
    N = len(S)

    # Condition: 1 <= i < j < k <= |S|
    # This implies a minimum length of 3 for S.
    if N < 3:
        print(0)
        return

    total_triples = 0

    # left_counts[char_code] stores the count of characters
    # in the substring S[0...j-1].
    # char_code 0 for 'A', 1 for 'B', ..., 25 for 'Z'.
    left_counts = [0] * 26

    # right_counts[char_code] stores the count of characters
    # in the substring S[j+1...N-1].
    right_counts = [0] * 26

    # Initialize right_counts for the substring S[1...N-1].
    # When j (0-based index of middle character) starts at 1,
    # S[0] is on the left, S[1] is the middle,
    # and S[2...N-1] forms the initial 'right' segment.
    # So, we populate right_counts with characters from S[1] onwards.
    for k in range(1, N):
        right_counts[ord(S[k]) - ord('A')] += 1

    # Iterate j from 1 to N-2 (inclusive).
    # These are the valid 0-based indices for the middle character S[j].
    # For a given j:
    #   i (0-based) must be in [0, j-1]
    #   k (0-based) must be in [j+1, N-1]
    for j in range(1, N - 1):
        # S[j-1] is the character that, in the context of the current 'j',
        # is now definitively part of the 'left' segment (S[0...j-1]).
        char_idx_prev_j = ord(S[j-1]) - ord('A')
        left_counts[char_idx_prev_j] += 1

        # S[j] is the current middle character.
        # It was previously counted in 'right_counts' (from S[j...N-1]).
        # Now that it's the middle element, it must be removed from 'right_counts'.
        char_idx_curr_j = ord(S[j]) - ord('A')
        right_counts[char_idx_curr_j] -= 1

        # Calculate the number of pairs (i, k) such that i < j < k and S[i] == S[k].
        # This is the sum of (left_counts[C] * right_counts[C]) for all characters C.
        current_j_contribution = 0
        for char_code in range(26):
            current_j_contribution += left_counts[char_code] * right_counts[char_code]
        
        total_triples += current_j_contribution
    
    print(total_triples)

solve()
import sys

def solve():
    n, k = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()

    # Length of the remaining string
    L = n - k 
    
    # Constraint: k < n implies L = n-k >= 1.
    # An empty string (L=0) would be a palindrome, but L >= 1 here.

    counts = [0] * 26
    for char_code in s:
        counts[ord(char_code) - ord('a')] += 1

    N_odd_initial = 0
    for i in range(26):
        if counts[i] % 2 != 0:
            N_odd_initial += 1
    
    # N_odd_target is the target number of character types with odd frequencies
    # in the remaining string for it to be a palindrome.
    # If L is even, N_odd_target must be 0.
    # If L is odd, N_odd_target must be 1.
    # This is equivalent to L % 2.
    N_odd_target = L % 2

    # k_parity_flips_needed is the minimum number of characters whose frequency parities
    # must be flipped to reach N_odd_target. Each such flip requires removing one character instance.
    k_parity_flips_needed = abs(N_odd_initial - N_odd_target)

    # Condition 1: Do we have enough removals for mandatory parity flips?
    if k < k_parity_flips_needed:
        sys.stdout.write("NO
")
        return

    # k_remaining_for_pairs are removals left after mandatory parity flips.
    # These must be used by removing pairs of identical characters to preserve N_odd_target.
    k_remaining_for_pairs = k - k_parity_flips_needed
    
    # Condition 2: Remaining removals must be used up in pairs, so k_remaining_for_pairs must be even.
    if k_remaining_for_pairs % 2 != 0:
        sys.stdout.write("NO
")
        return
    
    # After k_parity_flips_needed characters are removed (one for each flip):
    # - The string length becomes n_after_flips = n - k_parity_flips_needed.
    # - The number of character types with odd frequencies is N_odd_target.
    
    # The characters available for paired removal are those not "reserved" as one of
    # the N_odd_target odd-frequency characters (one instance of each such type).
    # Total count of characters that can form pairs = (n_after_flips - N_odd_target).
    # This value is guaranteed to be non-negative and even.
    chars_available_for_paired_removal_total_count = (n - k_parity_flips_needed) - N_odd_target
    
    # Condition 3: Do we have enough characters to perform the k_remaining_for_pairs removals?
    # These k_remaining_for_pairs removals must be taken from the pool of characters available for pairing.
    if k_remaining_for_pairs > chars_available_for_paired_removal_total_count:
        sys.stdout.write("NO
")
        return
        
    # All conditions met
    sys.stdout.write("YES
")

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()
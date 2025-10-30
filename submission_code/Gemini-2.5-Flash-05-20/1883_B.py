import sys
from collections import Counter

def solve():
    n, k = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()

    # Step 1: Count character frequencies in the original string
    char_counts = Counter(s)

    # Calculate initial_odd_count: number of distinct characters with odd frequencies
    initial_odd_count = 0
    for count in char_counts.values():
        if count % 2 != 0:
            initial_odd_count += 1

    # Calculate the length of the string after k removals
    remaining_length = n - k
    
    # Step 2: Determine target_odd_count for a palindrome of remaining_length
    # A palindrome of even length needs 0 odd-frequency chars.
    # A palindrome of odd length needs 1 odd-frequency char.
    target_odd_count = remaining_length % 2

    # Step 3: Calculate min_flips
    # This is the minimum number of character removals that must be used to change
    # the 'initial_odd_count' to 'target_odd_count'.
    # Each such removal flips the parity of one character's count.
    min_flips = abs(initial_odd_count - target_odd_count)

    # Step 4: Check feasibility conditions
    # Condition 1: We must have at least 'min_flips' removals.
    # If k is less than min_flips, we cannot even achieve the required odd/even character count state.
    if k < min_flips:
        sys.stdout.write("NO
")
        return

    # Condition 2: After using 'min_flips' removals to adjust the odd_count,
    # the remaining removals (k - min_flips) must be an even number.
    # If (k - min_flips) is odd, it means one more character's parity must be flipped,
    # which would then change the target_odd_count state (0 to 1, or 1 to 0).
    # Since we need to precisely hit 'target_odd_count', this is not allowed.
    if (k - min_flips) % 2 != 0:
        sys.stdout.write("NO
")
        return

    # If both conditions pass, it's possible.
    sys.stdout.write("YES
")


# Read the number of test cases
num_test_cases = int(sys.stdin.readline())

# Process each test case
for _ in range(num_test_cases):
    solve()
# YOUR CODE HERE
import sys
from collections import Counter

# Function to solve a single test case
def solve():
    # Read n (length of string) and k (number of characters to remove) from input
    n, k = map(int, sys.stdin.readline().split())
    # Read string s from input
    s = sys.stdin.readline().strip()
    
    # Count character frequencies using collections.Counter
    # This provides a dictionary-like object where keys are characters and values are their counts.
    # Example: Counter("aabbc") -> {'a': 2, 'b': 2, 'c': 1}
    counts = Counter(s)
    
    # Calculate the number of characters that appear an odd number of times in the original string s.
    # A character appears an odd number of times if its count is not divisible by 2.
    odd_counts_before = 0
    for char_count in counts.values():
        # Check if the count is odd
        if char_count % 2 != 0:
            odd_counts_before += 1
            
    # Calculate the length of the string after removing exactly k characters.
    remaining_len = n - k
    
    # A string can be rearranged into a palindrome if and only if at most one character appears an odd number of times.
    # The exact condition depends on the length of the string:
    # - If the string length is even, all characters must appear an even number of times (0 odd counts).
    # - If the string length is odd, exactly one character must appear an odd number of times (1 odd count).
    # We determine the target number of characters with odd counts required for the remaining string (of length `remaining_len`)
    # based on the parity of `remaining_len`.
    # `remaining_len % 2` yields 0 if even, 1 if odd, which matches the target number of odd counts.
    target_odd_counts = remaining_len % 2 
    
    # To transform the initial character counts structure (which has `odd_counts_before` characters with odd counts)
    # into the target structure (which requires `target_odd_counts` characters with odd counts),
    # we might need to change the parity of some character counts.
    # Changing the parity of a character's count (from odd to even, or even to odd) requires removing at least one occurrence of that character.
    # The minimum number of such parity flips required is the absolute difference between the initial number of odd counts (`odd_counts_before`)
    # and the target number of odd counts (`target_odd_counts`).
    # Therefore, the minimum number of characters that must be removed to potentially satisfy the palindrome property condition
    # is this absolute difference. Let's call this `min_removals_needed`.
    min_removals_needed = abs(odd_counts_before - target_odd_counts)

    # We are required to remove exactly k characters.
    # For it to be possible to end up with a string that can form a palindrome,
    # k must be at least the minimum number of removals needed to achieve the correct parity structure (`min_removals_needed`).
    # If k < min_removals_needed, it's impossible because we cannot even achieve the required parity structure with only k removals.
    # If k >= min_removals_needed, it has been established through combinatorial arguments that it is always possible.
    # This possibility relies on the fact that the difference `k - min_removals_needed` represents the number of additional characters
    # that must be removed after performing the minimum necessary removals to fix parities.
    # This difference (`k - min_removals_needed`) is always non-negative (since k >= min_removals_needed) and can be proven to be always even.
    # An even number of additional characters can always be removed by taking pairs of identical characters.
    # As long as k < n (guaranteed by problem constraints), there are sufficient characters available to remove these pairs.
    if k >= min_removals_needed:
        # If k is sufficient, print YES
        print("YES")
    else:
        # Otherwise, print NO
        print("NO")

# Read the total number of test cases
t = int(sys.stdin.readline())
# Iterate through each test case and call the solve function
for _ in range(t):
    solve()
import sys
from collections import Counter

# Read the number of test cases
t = int(sys.stdin.readline())

for _ in range(t):
    # Read n and k
    n, k = map(int, sys.stdin.readline().split())
    # Read the string s
    s = sys.stdin.readline().strip()

    # Count character frequencies
    freqs = Counter(s)

    # Count the number of characters with odd frequencies in the original string
    odd_count = 0
    for count in freqs.values():
        if count % 2 != 0:
            odd_count += 1

    # A string can be rearranged into a palindrome if and only if at most one
    # character appears an odd number of times.
    # We want the remaining string (of length n - k) to satisfy this property.
    # To reduce the number of character types with odd frequencies from `odd_count`
    # down to at most 1, we must change the parity of at least `odd_count - 1`
    # character types that initially had odd frequencies.
    # Changing the parity of a character type's frequency requires removing
    # an odd number of instances of that character type. The minimum number
    # of removals to change the parity for one character type is 1.
    # Thus, the minimum number of characters we *must* remove just to potentially
    # satisfy the palindrome property (by reducing the number of odd counts to at most 1)
    # is max(0, odd_count - 1).
    min_removals_for_palindrome_property = max(0, odd_count - 1)

    # We are required to remove exactly k characters.
    # If the number of characters we are allowed to remove (k) is less than
    # the minimum number of removals required to fix the parity issue,
    # then it's impossible to end up with at most one character type having
    # an odd frequency in the remaining string.
    if k < min_removals_for_palindrome_property:
        print("NO")
    else:
        # If k is greater than or equal to the minimum required removals
        # to fix the parity issue, we have enough removals available.
        # We can dedicate `min_removals_for_palindrome_property` removals
        # to change the parity of `min_removals_for_palindrome_property`
        # distinct character types that initially had odd frequencies (by
        # removing one instance of each). This results in a string with
        # at most one character type having an odd frequency.
        # We then have `k - min_removals_for_palindrome_property` additional
        # characters that must be removed. Since k >= min_removals_for_palindrome_property,
        # this number is non-negative. Since k < n, the length of the final
        # string `n - k` is > 0. We can always remove these additional characters
        # from the string of length `n - min_removals_for_palindrome_property`
        # to reach the target length `n - k`. As long as we have sufficient
        # removals (`k >= min_removals_for_palindrome_property`), it is possible
        # to achieve a state where the remaining `n-k` characters can form a palindrome.
        print("YES")
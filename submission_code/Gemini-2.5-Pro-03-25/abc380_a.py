import sys
from collections import Counter

def solve():
    # Read the 6-digit integer N as a string
    n_str = sys.stdin.readline().strip()

    # Check if the length is exactly 6 (guaranteed by constraints, but good practice)
    if len(n_str) != 6:
        # This case shouldn't happen based on constraints, but handle defensively
        print("No")
        return

    # Count the occurrences of each digit using collections.Counter
    digit_counts = Counter(n_str)

    # Check if the counts match the required conditions
    # Use .get(key, 0) to handle cases where a digit might not be present at all
    count1 = digit_counts.get('1', 0)
    count2 = digit_counts.get('2', 0)
    count3 = digit_counts.get('3', 0)

    # Verify all conditions
    if count1 == 1 and count2 == 2 and count3 == 3:
        # An additional implicit check: 1 + 2 + 3 = 6.
        # Since the string length is 6, if these counts are met,
        # no other digits can be present.
        print("Yes")
    else:
        print("No")

# Call the solve function to execute the logic
solve()
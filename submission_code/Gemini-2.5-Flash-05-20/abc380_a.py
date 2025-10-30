import sys
from collections import Counter

def solve():
    n_str = sys.stdin.readline().strip()

    # Constraints ensure N is a 6-digit number, so n_str will have length 6.
    # We can use collections.Counter to efficiently count digit occurrences.
    digit_counts = Counter(n_str)

    # Check the conditions for each digit:
    # - Digit '1' must appear exactly once.
    # - Digit '2' must appear exactly twice.
    # - Digit '3' must appear exactly three times.
    
    # Counter returns 0 for keys not present, which is convenient.
    count_1 = digit_counts['1']
    count_2 = digit_counts['2']
    count_3 = digit_counts['3']

    if count_1 == 1 and count_2 == 2 and count_3 == 3:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

if __name__ == '__main__':
    solve()
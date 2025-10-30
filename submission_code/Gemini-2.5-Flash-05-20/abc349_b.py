import sys
from collections import Counter

def solve():
    S = sys.stdin.readline().strip()

    # Step 1: Count the frequency of each character in S
    # Example: S = "commencement" -> {'c': 2, 'o': 1, 'm': 3, 'e': 3, 'n': 2, 't': 1}
    char_counts = Counter(S)

    # Step 2: Get the list of these character frequencies (how many times each distinct char appeared)
    # Example: From above, this would be [2, 1, 3, 3, 2, 1] (order may vary)
    counts_of_char_frequencies = list(char_counts.values())

    # Step 3: Count the occurrences of each frequency value.
    # This tells us, for each specific frequency 'i' (e.g., 1, 2, 3),
    # how many *different letters* appeared 'i' times.
    # Example: [2, 1, 3, 3, 2, 1] -> Counter({1: 2, 2: 2, 3: 2})
    # This means:
    # - For frequency 1: 2 different letters (o, t) appeared 1 time.
    # - For frequency 2: 2 different letters (c, n) appeared 2 times.
    # - For frequency 3: 2 different letters (m, e) appeared 3 times.
    frequency_distribution = Counter(counts_of_char_frequencies)

    # Step 4: Check the condition.
    # The problem states: "There are exactly zero or exactly two different letters that appear exactly i times in S."
    # If a frequency 'i' is not a key in frequency_distribution, it means 0 letters appear 'i' times,
    # which satisfies the "zero or two" condition.
    # If a frequency 'i' *is* a key, then the value frequency_distribution[i]
    # (which is the number of distinct characters appearing 'i' times) must be exactly 2.
    for num_distinct_chars_with_this_freq in frequency_distribution.values():
        if num_distinct_chars_with_this_freq != 2:
            print("No")
            return

    # If we go through all frequencies and none violate the condition, the string is good.
    print("Yes")

# Call the solve function to run the program
solve()
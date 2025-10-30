def can_win(S, T):
    from collections import Counter

    # Count characters in S and T
    count_S = Counter(S)
    count_T = Counter(T)

    # Count the number of '@' in both strings
    at_S = count_S['@']
    at_T = count_T['@']

    # Remove '@' from the counts
    del count_S['@']
    del count_T['@']

    # Check if we can match the counts of characters
    for char in count_S:
        if count_S[char] > count_T.get(char, 0) + at_S:
            return "No"

    for char in count_T:
        if count_T[char] > count_S.get(char, 0) + at_T:
            return "No"

    # Calculate the total number of '@' we can use
    total_at_available = at_S + at_T

    # Calculate how many replacements we need
    needed_replacements = 0
    for char in count_S:
        needed_replacements += max(0, count_S[char] - count_T.get(char, 0))

    for char in count_T:
        needed_replacements += max(0, count_T[char] - count_S.get(char, 0))

    # If we have enough '@' to cover the needed replacements, we can win
    if needed_replacements <= total_at_available:
        return "Yes"
    else:
        return "No"

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
S = data[0]
T = data[1]

# Output the result
print(can_win(S, T))
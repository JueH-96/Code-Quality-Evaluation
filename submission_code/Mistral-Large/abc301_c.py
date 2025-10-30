import sys
from collections import Counter

def can_win(S, T):
    # Replace '@' with an empty string in both S and T
    S_clean = S.replace('@', '')
    T_clean = T.replace('@', '')

    # Count the frequency of each character in the cleaned strings
    counter_S = Counter(S_clean)
    counter_T = Counter(T_clean)

    # Check if the counts of each character match
    for char in 'atcoder':
        if counter_S[char] != counter_T[char]:
            return "No"

    return "Yes"

# Read input
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Determine if it's possible to win and print the result
result = can_win(S, T)
print(result)
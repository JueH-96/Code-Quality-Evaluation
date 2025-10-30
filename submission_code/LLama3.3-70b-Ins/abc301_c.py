from collections import Counter

def solve(S, T):
    # Count the number of @s in each string
    at_S = S.count('@')
    at_T = T.count('@')

    # If the number of @s is not equal, it's impossible to win
    if at_S != at_T:
        return "No"

    # Count the frequency of each character in both strings
    count_S = Counter(S.replace('@', ''))
    count_T = Counter(T.replace('@', ''))

    # If the frequency of any character is not equal, it's impossible to win
    for char in set(S.replace('@', '') + T.replace('@', '')):
        if count_S[char] + at_S < count_T[char] or count_T[char] + at_T < count_S[char]:
            return "No"

    return "Yes"

# Read input from stdin
S = input()
T = input()

# Solve the problem and print the answer
print(solve(S, T))
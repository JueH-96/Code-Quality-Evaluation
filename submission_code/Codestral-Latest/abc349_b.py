from collections import Counter

def is_good_string(S):
    # Count the frequency of each character in the string
    freq = Counter(S)

    # Count the frequency of each frequency
    freq_of_freq = Counter(freq.values())

    # Check if the frequency of frequencies satisfies the condition
    for count in freq_of_freq.values():
        if count != 0 and count != 2:
            return False

    return True

# Read input from stdin
S = input().strip()

# Determine if the string is a good string
if is_good_string(S):
    print("Yes")
else:
    print("No")
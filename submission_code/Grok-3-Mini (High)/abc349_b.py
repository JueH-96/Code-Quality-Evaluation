import collections

# Read the input string from standard input
S = input().strip()

# Count the frequency of each letter in the string
letter_count = collections.Counter(S)

# Count how many letters have each frequency
freq_count = collections.Counter(letter_count.values())

# Check if for every frequency that appears, exactly two letters have that frequency
if all(count == 2 for count in freq_count.values()):
    print("Yes")
else:
    print("No")
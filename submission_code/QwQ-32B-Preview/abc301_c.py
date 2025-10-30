allowed = 'atcoder'

# Read S and T
S = input()
T = input()

# Count frequencies
from collections import Counter
count_S = Counter(S)
count_T = Counter(T)

# Check letters not in allowed
for letter in count_S:
    if letter not in allowed and letter != '@':
        if count_S[letter] != count_T[letter]:
            print("No")
            exit()

for letter in count_T:
    if letter not in allowed and letter != '@':
        if count_T[letter] != count_S[letter]:
            print("No")
            exit()

# Calculate required @ for S and T
required_S = 0
required_T = 0
for letter in allowed:
    diff = count_S[letter] - count_T[letter]
    if diff > 0:
        required_T += diff
    elif diff < 0:
        required_S -= diff

# Check if enough @
if required_S <= count_S['@'] and required_T <= count_T['@']:
    print("Yes")
else:
    print("No")
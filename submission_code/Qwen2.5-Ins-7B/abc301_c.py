# YOUR CODE HERE
from collections import Counter

S = input().strip()
T = input().strip()

sc = Counter(S)
tc = Counter(T)

# Count the number of '@' in each string
at_count_s = S.count('@')
at_count_t = T.count('@')

# Count the number of each character in S and T excluding '@'
sc_excl = sc - tc
tc_excl = tc - sc

# Check if it's possible to replace '@' to make S and T equal
if sc_excl == tc_excl and at_count_s + at_count_t >= abs(sc_excl - tc_excl):
    print('Yes')
else:
    print('No')
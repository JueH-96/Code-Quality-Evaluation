# YOUR CODE HERE
S = input()
from collections import Counter
count = Counter(S)

frequencies = count.values()
max_freq = max(frequencies)

is_good = True
for i in range(1, max_freq+1):
    letters_with_i = sum(1 for freq in frequencies if freq == i)
    if letters_with_i != 0 and letters_with_i != 2:
        is_good = False
        break

print('Yes' if is_good else 'No')
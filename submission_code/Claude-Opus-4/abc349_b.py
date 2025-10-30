# YOUR CODE HERE
from collections import Counter

S = input().strip()

# Count frequency of each letter
freq_count = Counter(S)

# Get the frequency of each letter
frequencies = list(freq_count.values())

# Count how many letters appear exactly i times for each i
freq_of_freq = Counter(frequencies)

# Check if it's a good string
is_good = True
for count in freq_of_freq.values():
    if count != 0 and count != 2:
        is_good = False
        break

if is_good:
    print("Yes")
else:
    print("No")
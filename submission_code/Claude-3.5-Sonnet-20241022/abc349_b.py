from collections import Counter

# Read input
S = input().strip()

# Count frequency of each letter
freq = Counter(S)

# Count how many letters appear i times
freq_count = Counter(freq.values())

# Check if for each frequency i, there are exactly 0 or 2 letters
is_good = True
for i in range(1, max(freq.values()) + 1):
    if freq_count[i] != 0 and freq_count[i] != 2:
        is_good = False
        break

# Print result
print("Yes" if is_good else "No")
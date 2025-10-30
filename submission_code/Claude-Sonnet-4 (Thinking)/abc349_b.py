s = input()

# Count frequency of each character
char_freq = {}
for char in s:
    char_freq[char] = char_freq.get(char, 0) + 1

# Count how many letters appear exactly i times
freq_count = {}
for freq in char_freq.values():
    freq_count[freq] = freq_count.get(freq, 0) + 1

# Check if all counts are 2 (can't be 0 since we only count existing frequencies)
is_good = all(count == 2 for count in freq_count.values())

print("Yes" if is_good else "No")
# YOUR CODE HERE
s = input().strip()

# Count frequency of each character
freq_count = {}
for char in s:
    freq_count[char] = freq_count.get(char, 0) + 1

# Count how many letters appear exactly i times for each frequency i
frequency_distribution = {}
for char, freq in freq_count.items():
    frequency_distribution[freq] = frequency_distribution.get(freq, 0) + 1

# Check if the string is good
is_good = True
max_freq = max(freq_count.values()) if freq_count else 0

for i in range(1, max_freq + 1):
    count = frequency_distribution.get(i, 0)
    if count != 0 and count != 2:
        is_good = False
        break

print("Yes" if is_good else "No")
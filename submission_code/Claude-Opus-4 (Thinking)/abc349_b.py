# YOUR CODE HERE
s = input().strip()

# Count frequency of each letter
freq = {}
for char in s:
    freq[char] = freq.get(char, 0) + 1

# Count how many letters have each frequency
freq_count = {}
for count in freq.values():
    freq_count[count] = freq_count.get(count, 0) + 1

# Check if all counts are 2 (they can't be 0 since they exist in freq_count)
is_good = all(count == 2 for count in freq_count.values())

if is_good:
    print("Yes")
else:
    print("No")
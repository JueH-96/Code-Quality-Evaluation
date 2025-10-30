s = input()
freq = {}
for char in s:
    freq[char] = freq.get(char, 0) + 1

max_freq = max(freq.values())
result = min(char for char, count in freq.items() if count == max_freq)
print(result)
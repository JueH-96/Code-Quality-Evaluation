from collections import Counter

s = input()
freq_counter = Counter(s)
max_freq = 0
for count in freq_counter.values():
    max_freq = max(max_freq, count)

result_char = None
for char in sorted(freq_counter.keys()):
    if freq_counter[char] == max_freq:
        result_char = char
        break
print(result_char)
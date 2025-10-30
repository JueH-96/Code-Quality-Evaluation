from collections import Counter

s = input().strip()
freq = Counter(s)
max_freq = max(freq.values())

for char in 'abcdefghijklmnopqrstuvwxyz':
    if freq[char] == max_freq:
        print(char)
        break
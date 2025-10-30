s = input().strip()
freq = [0] * 26
for c in s:
    freq[ord(c) - ord('a')] += 1
max_freq = max(freq)
for i in range(26):
    if freq[i] == max_freq:
        print(chr(ord('a') + i))
        break
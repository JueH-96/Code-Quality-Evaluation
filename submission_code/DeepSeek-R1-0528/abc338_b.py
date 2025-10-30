s = input().strip()
freq = [0] * 26

for char in s:
    idx = ord(char) - ord('a')
    freq[idx] += 1

max_count = max(freq)

for i in range(26):
    if freq[i] == max_count:
        print(chr(ord('a') + i))
        break
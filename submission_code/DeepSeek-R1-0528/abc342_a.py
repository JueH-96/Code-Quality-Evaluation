s = input().strip()
freq = {}
for char in s:
    freq[char] = freq.get(char, 0) + 1

for idx, char in enumerate(s):
    if freq[char] == 1:
        print(idx + 1)
        break
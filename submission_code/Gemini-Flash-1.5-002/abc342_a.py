# YOUR CODE HERE
s = input()
counts = {}
for char in s:
    counts[char] = counts.get(char, 0) + 1

diff_char = ''
same_char = ''
for char, count in counts.items():
    if count == 1:
        diff_char = char
    else:
        same_char = char

for i in range(len(s)):
    if s[i] == diff_char:
        print(i + 1)
        break
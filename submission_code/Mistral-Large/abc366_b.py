import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
strings = data[1:]

max_len = max(len(s) for s in strings)
vertical_strings = ['' for _ in range(max_len)]

for i in range(N):
    for j in range(len(strings[i])):
        if j < len(vertical_strings) - len(strings[i]):
            vertical_strings[j] += '*'
        else:
            vertical_strings[j] += strings[i][j]

for i in range(max_len):
    vertical_strings[i] = vertical_strings[i][::-1]
    if vertical_strings[i].endswith('*'):
        vertical_strings[i] = vertical_strings[i].rstrip('*')

print("
".join(vertical_strings))
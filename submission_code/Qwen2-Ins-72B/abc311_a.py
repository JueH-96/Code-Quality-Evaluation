N = int(input())
S = input()

chars = {'A', 'B', 'C'}
found_chars = set()
count = 0

for i in range(N):
    found_chars.add(S[i])
    if len(found_chars) == 3:
        count = i + 1
        break

print(count)
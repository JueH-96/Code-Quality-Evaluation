n = int(input())
s = input().strip()
q = int(input())

current_map = [chr(ord('a') + i) for i in range(26)]

for _ in range(q):
    c, d = input().split()
    for x in range(26):
        if current_map[x] == c:
            current_map[x] = d

result = []
for char in s:
    result.append(current_map[ord(char) - ord('a')])

print(''.join(result))
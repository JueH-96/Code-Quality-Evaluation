n = int(input())
s = input().strip()
q = int(input())

char_map = [chr(ord('a') + i) for i in range(26)]

for _ in range(q):
    c, d = input().split()
    c = c[0]
    d = d[0]
    for i in range(26):
        if char_map[i] == c:
            char_map[i] = d

result = ''.join([char_map[ord(c) - ord('a')] for c in s])
print(result)
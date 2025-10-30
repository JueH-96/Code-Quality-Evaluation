n, d = map(int, input().split())
s = input().strip()

positions = [i for i, c in enumerate(s) if c == '@']
removed = set(positions[-d:]) if d > 0 else set()

result = []
for i in range(n):
    if s[i] == '@' and i not in removed:
        result.append('@')
    else:
        result.append('.')
print(''.join(result))
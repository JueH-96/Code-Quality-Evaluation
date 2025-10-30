n, d = map(int, input().split())
s = input().strip()

cookies = []
for i in range(n):
    if s[i] == '@':
        cookies.append(i + 1)  # Storing 1-based positions

total_cookies = len(cookies)
kept_num = total_cookies - d

kept_positions = set(cookies[:kept_num])

result = []
for i in range(n):
    pos = i + 1
    if pos in kept_positions:
        result.append('@')
    else:
        result.append('.')

print(''.join(result))
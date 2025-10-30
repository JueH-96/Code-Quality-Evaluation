n, d = map(int, input().split())
s = input().strip()
count_dot = s.count('.')
result = count_dot + d
print(result)
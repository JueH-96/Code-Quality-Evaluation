n, d = map(int, input().split())
s = input().strip()
count = s.count('@')
empty = (n - (count - d)) - 1
print(empty)
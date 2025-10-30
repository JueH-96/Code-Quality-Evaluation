n, d = map(int, input().split())
s = input().strip()
count = s.count('@')
print(n - count + d)
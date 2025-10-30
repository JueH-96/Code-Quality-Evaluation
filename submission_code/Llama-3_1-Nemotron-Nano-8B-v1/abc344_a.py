s = input().strip()
first = s.index('|')
second = s.index('|', first + 1)
print(s[:first] + s[second+1:])
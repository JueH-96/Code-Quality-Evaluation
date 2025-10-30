s = input().strip()
first = s.find('|')
second = s.find('|', first + 1)
print(s[:first] + s[second + 1:])
s = input().strip()
first = s.find('|')
second = s.find('|', first + 1)
result = s[:first] + s[second+1:]
print(result)
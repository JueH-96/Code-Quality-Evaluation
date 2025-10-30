s = input().strip()
start = s.find('|')
end = s.find('|', start + 1)
result = s[:start] + s[end+1:]
print(result)
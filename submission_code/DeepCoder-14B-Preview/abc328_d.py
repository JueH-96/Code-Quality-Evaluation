s = input().strip()
while True:
    pos = s.find('ABC')
    if pos == -1:
        break
    s = s[:pos] + s[pos+3:]
print(s)
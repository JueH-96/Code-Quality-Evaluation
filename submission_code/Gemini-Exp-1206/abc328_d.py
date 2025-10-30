s = input()
while 'ABC' in s:
    s = s.replace('ABC', '', 1)
print(s)
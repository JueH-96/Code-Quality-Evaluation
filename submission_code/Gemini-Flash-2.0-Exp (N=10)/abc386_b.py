s = input()
count = 0
i = 0
while i < len(s):
    if i + 1 < len(s) and s[i] == '0' and s[i+1] == '0':
        count += 1
        i += 2
    else:
        count += 1
        i += 1
print(count)
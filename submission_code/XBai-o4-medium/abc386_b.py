s = input().strip()
i = 0
count = 0
while i < len(s):
    if s[i] != '0':
        count += 1
        i += 1
    else:
        if i + 1 < len(s) and s[i+1] == '0':
            count += 1
            i += 2
        else:
            count += 1
            i += 1
print(count)
n = int(input())
s = input().strip()
t = input().strip()

result = "Yes"
for sc, tc in zip(s, t):
    if sc == tc:
        continue
    elif (sc == '1' and tc == 'l') or (sc == 'l' and tc == '1'):
        continue
    elif (sc == '0' and tc == 'o') or (sc == 'o' and tc == '0'):
        continue
    else:
        result = "No"
        break
print(result)
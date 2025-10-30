s = input().strip()
total = 0
i = 0
n = len(s)

while i < n:
    if s[i] != '0':
        total += 1
        i += 1
    else:
        j = i
        while j < n and s[j] == '0':
            j += 1
        cnt = j - i
        total += (cnt + 1) // 2
        i = j

print(total)
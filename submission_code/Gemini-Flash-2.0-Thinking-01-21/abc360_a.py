s = input()
index_r = -1
index_m = -1
for i in range(len(s)):
    if s[i] == 'R':
        index_r = i
    elif s[i] == 'M':
        index_m = i

if index_r < index_m:
    print("Yes")
else:
    print("No")
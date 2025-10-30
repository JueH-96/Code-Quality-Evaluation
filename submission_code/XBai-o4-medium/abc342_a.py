s = input().strip()

if s[0] == s[1]:
    majority = s[0]
else:
    if s[0] == s[2]:
        majority = s[0]
    else:
        majority = s[1]

for i in range(len(s)):
    if s[i] != majority:
        print(i + 1)
        break
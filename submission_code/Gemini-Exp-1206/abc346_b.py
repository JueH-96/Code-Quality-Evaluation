w, b = map(int, input().split())
s = "wbwbwwbwbwbw" * 20
for i in range(len(s)):
    for j in range(i, len(s)):
        sub = s[i : j + 1]
        if sub.count('w') == w and sub.count('b') == b:
            print("Yes")
            exit()
print("No")
s = input().strip()
for i in range(1, 16, 2):
    if s[i] != '0':
        print("No")
        exit()
print("Yes")
s = input()
for i in range(2, 17, 2):
    if s[i-1] != '0':
        print("No")
        exit()
print("Yes")
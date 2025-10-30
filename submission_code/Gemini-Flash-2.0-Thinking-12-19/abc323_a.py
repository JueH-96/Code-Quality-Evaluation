s = input()
for i in range(1, 17):
    if i % 2 == 0:
        if s[i-1] != '0':
            print("No")
            exit()
print("Yes")
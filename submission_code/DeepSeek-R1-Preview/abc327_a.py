n = int(input())
s = input().strip()

for i in range(n - 1):
    if (s[i] == 'a' and s[i+1] == 'b') or (s[i] == 'b' and s[i+1] == 'a'):
        print("Yes")
        exit()
print("No")
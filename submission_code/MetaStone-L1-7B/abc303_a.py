def is_similar(a, b):
    if a == b:
        return True
    if (a == 'l' and b == '1') or (a == '1' and b == 'l'):
        return True
    if (a == '0' and b == 'o') or (a == 'o' and b == '0'):
        return True
    return False

n = int(input())
s = input().strip()
t = input().strip()

for i in range(n):
    if not is_similar(s[i], t[i]):
        print("No")
        exit()

print("Yes")
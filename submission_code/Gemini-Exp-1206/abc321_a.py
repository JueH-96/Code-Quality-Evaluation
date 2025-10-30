def is_321_like(n):
    s = str(n)
    for i in range(len(s) - 1):
        if int(s[i]) <= int(s[i+1]):
            return False
    return True

n = int(input())
if is_321_like(n):
    print("Yes")
else:
    print("No")
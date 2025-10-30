s = list(map(int, input().split()))

def solve():
    for i in range(8):
        if not (100 <= s[i] <= 675):
            return False
        if s[i] % 25 != 0:
            return False
    for i in range(7):
        if s[i] > s[i+1]:
            return False
    return True

if solve():
    print("Yes")
else:
    print("No")
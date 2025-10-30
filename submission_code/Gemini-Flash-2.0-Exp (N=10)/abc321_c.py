def is_321_like(n):
    s = str(n)
    for i in range(len(s) - 1):
        if int(s[i]) <= int(s[i+1]):
            return False
    return True

def solve():
    k = int(input())
    count = 0
    num = 0
    while True:
        if is_321_like(num):
            count += 1
            if count == k:
                print(num)
                return
        num += 1

solve()
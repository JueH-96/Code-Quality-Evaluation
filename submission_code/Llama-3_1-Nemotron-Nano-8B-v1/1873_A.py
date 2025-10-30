t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
    else:
        swapped1 = s[1] + s[0] + s[2]
        swapped2 = s[2] + s[1] + s[0]
        swapped3 = s[0] + s[2] + s[1]
        if swapped1 == 'abc' or swapped2 == 'abc' or swapped3 == 'abc':
            print("YES")
        else:
            print("NO")
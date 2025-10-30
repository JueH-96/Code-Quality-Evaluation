t = int(input())
for _ in range(t):
    s = input().strip()
    if s == 'abc':
        print("YES")
    else:
        swap1 = s[1] + s[0] + s[2]
        swap2 = s[2] + s[1] + s[0]
        swap3 = s[0] + s[2] + s[1]
        if swap1 == 'abc' or swap2 == 'abc' or swap3 == 'abc':
            print("YES")
        else:
            print("NO")
def solve():
    n = int(input())
    s = input()

    found = False
    for i in range(n - 1):
        if (s[i] == 'a' and s[i+1] == 'b') or (s[i] == 'b' and s[i+1] == 'a'):
            found = True
            break

    if found:
        print("Yes")
    else:
        print("No")

solve()
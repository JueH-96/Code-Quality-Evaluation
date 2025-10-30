def solve():
    k = int(input())
    s = input()
    t = input()

    if s == t:
        print("Yes")
        return

    if k == 0:
        print("No")
        return

    n = len(s)
    m = len(t)

    # Insertion
    for i in range(n + 1):
        for char in "abcdefghijklmnopqrstuvwxyz":
            temp = s[:i] + char + s[i:]
            if temp == t:
                print("Yes")
                return

    # Deletion
    for i in range(n):
        temp = s[:i] + s[i+1:]
        if temp == t:
            print("Yes")
            return

    # Replacement
    for i in range(n):
        for char in "abcdefghijklmnopqrstuvwxyz":
            if s[i] != char:
                temp = s[:i] + char + s[i+1:]
                if temp == t:
                    print("Yes")
                    return
    
    print("No")

solve()
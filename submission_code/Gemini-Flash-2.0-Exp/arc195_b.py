def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    a_neg_count = a.count(-1)
    b_neg_count = b.count(-1)

    a_pos = [x for x in a if x != -1]
    b_pos = [x for x in b if x != -1]

    if len(a_pos) + a_neg_count != n or len(b_pos) + b_neg_count != n:
        print("No")
        return

    if n == 2:
        if a[0] == -1 and a[1] == -1:
            print("Yes")
            return
        if b[0] == -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] == -1 and b[0] == -1:
            print("Yes")
            return
        if a[1] == -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] == -1 and b[1] == -1:
            print("Yes")
            return
        if a[1] == -1 and b[0] == -1:
            print("Yes")
            return
        if a[0] != -1 and a[1] != -1 and b[0] != -1 and b[1] != -1:
            if a[0] + b[0] == a[1] + b[1]:
                print("Yes")
            else:
                print("No")
            return
        if a[0] != -1 and a[1] != -1 and b[0] == -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] == -1 and a[1] == -1 and b[0] != -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] != -1 and a[1] == -1 and b[0] != -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] == -1 and a[1] != -1 and b[0] != -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] != -1 and a[1] == -1 and b[0] == -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] == -1 and a[1] != -1 and b[0] == -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] != -1 and a[1] != -1 and b[0] != -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] != -1 and a[1] != -1 and b[0] == -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] == -1 and a[1] != -1 and b[0] != -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] != -1 and a[1] == -1 and b[0] != -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] != -1 and a[1] != -1 and b[0] != -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] != -1 and a[1] != -1 and b[0] == -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] != -1 and a[1] == -1 and b[0] != -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] == -1 and a[1] != -1 and b[0] != -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] != -1 and b[0] != -1 and a[1] == -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] == -1 and b[0] == -1 and a[1] != -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] != -1 and b[0] == -1 and a[1] == -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] == -1 and b[0] != -1 and a[1] != -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] != -1 and b[0] == -1 and a[1] != -1 and b[1] == -1:
            print("Yes")
            return
        if a[0] == -1 and b[0] != -1 and a[1] == -1 and b[1] != -1:
            print("Yes")
            return
        if a[0] != -1 and b[1] == -1 and a[1] == -1 and b[0] != -1:
            print("Yes")
            return
        if a[0] == -1 and b[1] != -1 and a[1] != -1 and b[0] == -1:
            print("Yes")
            return
        if a[0] != -1 and b[1] == -1 and a[1] != -1 and b[0] == -1:
            print("Yes")
            return
        if a[0] == -1 and b[1] != -1 and a[1] == -1 and b[0] != -1:
            print("Yes")
            return
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    m = int(input())
    b = list(map(int, input().split()))
    l = int(input())
    c = list(map(int, input().split()))
    q = int(input())
    x = list(map(int, input().split()))

    for target in x:
        found = False
        for val_a in a:
            for val_b in b:
                for val_c in c:
                    if val_a + val_b + val_c == target:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            print("Yes")
        else:
            print("No")

solve()
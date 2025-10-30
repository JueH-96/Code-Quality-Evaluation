def solve():
    n = int(input())
    a = list(map(int, input().split()))
    m = int(input())
    b = list(map(int, input().split()))
    l = int(input())
    c = list(map(int, input().split()))
    q = int(input())
    x = list(map(int, input().split()))

    ab_sums = set()
    for val_a in a:
        for val_b in b:
            ab_sums.add(val_a + val_b)

    for target in x:
        found = False
        for val_c in c:
            if target - val_c in ab_sums:
                found = True
                break
        if found:
            print("Yes")
        else:
            print("No")

solve()
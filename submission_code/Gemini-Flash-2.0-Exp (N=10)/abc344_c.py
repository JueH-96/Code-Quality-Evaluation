def solve():
    N = int(input())
    A = list(map(int, input().split()))
    M = int(input())
    B = list(map(int, input().split()))
    L = int(input())
    C = list(map(int, input().split()))
    Q = int(input())
    X = list(map(int, input().split()))

    for target in X:
        found = False
        for a in A:
            for b in B:
                for c in C:
                    if a + b + c == target:
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
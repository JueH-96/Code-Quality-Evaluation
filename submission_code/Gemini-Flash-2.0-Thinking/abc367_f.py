from collections import Counter

def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for _ in range(q):
        l, r, L, R = map(int, input().split())
        sub_a = a[l-1:r]
        sub_b = b[L-1:R]

        if len(sub_a) != len(sub_b):
            print("No")
            continue

        count_a = Counter(sub_a)
        count_b = Counter(sub_b)

        if count_a == count_b:
            print("Yes")
        else:
            print("No")

solve()
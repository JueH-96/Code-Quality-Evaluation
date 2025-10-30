# YOUR CODE HERE
def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for _ in range(q):
        l, r, ll, rr = map(int, input().split())
        sub_a = a[l-1:r]
        sub_b = b[ll-1:rr]

        if len(sub_a) != len(sub_b):
            print("No")
            continue

        sub_a.sort()
        sub_b.sort()

        if sub_a == sub_b:
            print("Yes")
        else:
            print("No")

solve()
n, q = map(int, input().split())
forbidden_low = n * (n - 1)
forbidden_high = n * n - 1

for _ in range(q):
    k = int(input())
    if k < forbidden_low or k > forbidden_high:
        print("Yes")
    else:
        print("No")
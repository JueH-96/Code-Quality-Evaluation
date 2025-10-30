# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_prod = 0
    for i in range(n):
        temp_a = list(a)
        temp_a[i] += 1
        prod = 1
        for x in temp_a:
            prod *= x
        max_prod = max(max_prod, prod)

    print(max_prod)


t = int(input())
for _ in range(t):
    solve()
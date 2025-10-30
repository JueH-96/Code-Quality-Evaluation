# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    b = []
    for i in range(n - 1):
        b.append(a[i] * a[i+1])

    print(*b)

solve()
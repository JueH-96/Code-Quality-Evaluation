from math import gcd

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def can_win(i):
        for j in range(1, a[i]):
            if a[i] % j == 0 and j != a[i]:
                return True
        return False

    if all(can_win(i) for i in range(n)):
        print("Anna")
    else:
        print("Bruno")

solve()
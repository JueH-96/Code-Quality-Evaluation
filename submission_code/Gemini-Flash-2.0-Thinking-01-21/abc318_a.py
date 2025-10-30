import math

def solve():
    n, m, p = map(int, input().split())
    if m > n:
        print(0)
    else:
        k_end = (n - m) // p
        if k_end < 0:
            print(0)
        else:
            print(k_end + 1)

if __name__ == '__main__':
    solve()
def solve():
    n, l, r = map(int, input().split())
    a = list(range(1, n + 1))
    l -= 1
    r -= 1
    sub = a[l:r+1]
    reversed_sub = sub[::-1]
    a[l:r+1] = reversed_sub
    print(*(a))

if __name__ == "__main__":
    solve()
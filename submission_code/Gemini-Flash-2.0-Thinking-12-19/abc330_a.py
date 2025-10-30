def solve():
    n, l = map(int, input().split())
    a = list(map(int, input().split()))

    pass_count = 0
    for score in a:
        if score >= l:
            pass_count += 1
    print(pass_count)

if __name__ == "__main__":
    solve()
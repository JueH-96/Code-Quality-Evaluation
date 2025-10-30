def solve():
    N, T, P = map(int, input().split())
    L = list(map(int, input().split()))

    count = sum(1 for length in L if length >= T)

    if count >= P:
        print(0)
        return

    days = 0
    while True:
        days += 1
        L = [length + 1 for length in L]
        count = sum(1 for length in L if length >= T)
        if count >= P:
            print(days)
            return

if __name__ == "__main__":
    solve()
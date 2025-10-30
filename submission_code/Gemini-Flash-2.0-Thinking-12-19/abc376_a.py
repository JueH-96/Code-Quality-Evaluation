def solve():
    n, c = map(int, input().split())
    t = list(map(int, input().split()))

    candy_count = 0
    last_candy_time = -c

    for current_time in t:
        if current_time - last_candy_time >= c:
            candy_count += 1
            last_candy_time = current_time

    print(candy_count)

if __name__ == "__main__":
    solve()
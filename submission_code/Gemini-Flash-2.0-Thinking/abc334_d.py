def solve():
    n, q = map(int, input().split())
    r = list(map(int, input().split()))
    r_sorted = sorted(r)

    for _ in range(q):
        x = int(input())
        count = 0
        remaining_reindeer = x
        for req in r_sorted:
            if remaining_reindeer >= req:
                remaining_reindeer -= req
                count += 1
            else:
                break
        print(count)

solve()
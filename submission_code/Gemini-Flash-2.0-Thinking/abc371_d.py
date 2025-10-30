import bisect

def solve():
    n = int(input())
    x = list(map(int, input().split()))
    p = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        l, r = map(int, input().split())
        total_villagers = 0

        start_index = bisect.bisect_left(x, l)
        end_index = bisect.bisect_right(x, r)

        for i in range(start_index, end_index):
            total_villagers += p[i]

        print(total_villagers)

solve()
from bisect import bisect_right

def solve():
    w, h = map(int, input().split())
    n = int(input())
    strawberries = []
    for _ in range(n):
        strawberries.append(tuple(map(int, input().split())))
    a_count = int(input())
    a = list(map(int, input().split()))
    b_count = int(input())
    b = list(map(int, input().split()))

    counts = {}

    for p, q in strawberries:
        x_index = bisect_right(sorted(a), p)
        y_index = bisect_right(sorted(b), q)

        if (x_index, y_index) not in counts:
            counts[(x_index, y_index)] = 0
        counts[(x_index, y_index)] += 1

    if not counts:
        print(0, 0)
    else:
        min_strawberries = min(counts.values())
        max_strawberries = max(counts.values())
        print(min_strawberries, max_strawberries)

solve()
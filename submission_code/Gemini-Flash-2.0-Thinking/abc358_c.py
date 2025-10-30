import itertools

def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]

    stand_flavors = []
    for i in range(n):
        flavors = set()
        for j in range(m):
            if s[i][j] == 'o':
                flavors.add(j + 1)
        stand_flavors.append(flavors)

    for num_stands in range(1, n + 1):
        for combination in itertools.combinations(range(n), num_stands):
            all_flavors = set()
            for stand_index in combination:
                all_flavors.update(stand_flavors[stand_index])

            if len(all_flavors) == m:
                print(num_stands)
                return

solve()
def solve():
    n = int(input())
    balls = []
    for _ in range(n):
        x, y = map(int, input().split())
        balls.append((x, y))

    def remove_balls(remaining_balls, k):
        new_remaining_balls = []
        for i in remaining_balls:
            if i != k:
                xi, yi = balls[i-1]
                xk, yk = balls[k-1]
                if not ((xi < xk and yi < yk) or (xi > xk and yi > yk)):
                    new_remaining_balls.append(i)
        new_remaining_balls.append(k)
        return new_remaining_balls

    possible_sets = set()

    def find_possible_sets(remaining_balls):
        remaining_balls = tuple(sorted(remaining_balls))
        if remaining_balls in possible_sets:
            return
        possible_sets.add(remaining_balls)

        for k in list(remaining_balls):
            new_remaining_balls = remove_balls(list(remaining_balls), k)
            find_possible_sets(new_remaining_balls)

    find_possible_sets(list(range(1, n + 1)))
    print(len(possible_sets) % 998244353)

solve()
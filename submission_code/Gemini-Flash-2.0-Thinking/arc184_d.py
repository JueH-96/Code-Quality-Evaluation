def solve():
    N = int(input())
    coords = []
    for _ in range(N):
        x, y = map(int, input().split())
        coords.append((x, y))

    X = [c[0] for c in coords]
    Y = [c[1] for c in coords]

    def is_antichain(subset_indices):
        for i_idx in range(len(subset_indices)):
            for j_idx in range(i_idx + 1, len(subset_indices)):
                b1 = subset_indices[i_idx]
                b2 = subset_indices[j_idx]

                if (X[b2] < X[b1] and Y[b2] < Y[b1]) or (X[b2] > X[b1] and Y[b2] > Y[b1]):
                    return False
        return True

    possible_remaining_sets = set()

    def find_remaining_sets(current_balls):
        if tuple(sorted(current_balls)) not in possible_remaining_sets:
            possible_remaining_sets.add(tuple(sorted(current_balls)))
            if not current_balls:
                return

            for k_idx in range(len(current_balls)):
                k = current_balls[k_idx]

                next_balls = list(current_balls)
                balls_to_remove = set()
                for i_idx in range(len(next_balls)):
                    i = next_balls[i_idx]
                    if i == k:
                        continue

                    if (X[i] < X[k] and Y[i] < Y[k]) or (X[i] > X[k] and Y[i] > Y[k]):
                        balls_to_remove.add(i)

                new_remaining_balls = set(next_balls) - balls_to_remove
                find_remaining_sets(tuple(sorted(list(new_remaining_balls))))

    initial_balls = tuple(range(N))
    find_remaining_sets(initial_balls)

    print(len(possible_remaining_sets) % 998244353)

solve()
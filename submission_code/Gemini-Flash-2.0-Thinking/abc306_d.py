def solve():
    n = int(input())
    courses = []
    for _ in range(n):
        x, y = map(int, input().split())
        courses.append((x, y))

    dp_healthy = 0
    dp_upset = -float('inf')

    for x, y in courses:
        new_dp_healthy = dp_healthy
        new_dp_upset = dp_upset

        # Consider skipping the current course:
        next_dp_healthy = new_dp_healthy
        next_dp_upset = new_dp_upset

        # Consider eating the current course:
        if x == 0:  # Antidotal
            next_dp_healthy = max(next_dp_healthy, dp_healthy + y)
            next_dp_healthy = max(next_dp_healthy, dp_upset + y)
        elif x == 1:  # Poisonous
            next_dp_upset = max(next_dp_upset, dp_healthy + y)

        dp_healthy = next_dp_healthy
        dp_upset = next_dp_upset

    print(max(0, dp_healthy, dp_upset))

solve()
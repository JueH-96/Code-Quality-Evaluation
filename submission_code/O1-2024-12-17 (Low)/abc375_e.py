def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = []
    B = []
    idx = 1
    for _ in range(N):
        A.append(int(input_data[idx]))
        B.append(int(input_data[idx+1]))
        idx += 2

    total_strength = sum(B)
    # If total is not divisible by 3, it's impossible
    if total_strength % 3 != 0:
        print(-1)
        return

    target = total_strength // 3

    # Prefix sums for pruning: prefix_sum[i] = sum of B[:i]
    prefix_sum = [0] * (N+1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + B[i]

    # dp will map (s1, s2) -> minimum cost (number of moves)
    # where s1 is total strength assigned to team1, s2 is total strength to team2
    # team3 is then prefix_sum[i] - s1 - s2 for the first i people.
    dp = {(0, 0): 0}

    for i in range(N):
        new_dp = {}
        strength = B[i]
        old_team = A[i]
        for (s1, s2), cost_so_far in dp.items():
            # Put person i in team1
            ns1, ns2 = s1 + strength, s2
            if ns1 <= target and ns2 <= target and ns1 + ns2 <= prefix_sum[i+1]:
                new_cost = cost_so_far + (1 if old_team != 1 else 0)
                if (ns1, ns2) not in new_dp or new_dp[(ns1, ns2)] > new_cost:
                    new_dp[(ns1, ns2)] = new_cost

            # Put person i in team2
            ns1, ns2 = s1, s2 + strength
            if ns1 <= target and ns2 <= target and ns1 + ns2 <= prefix_sum[i+1]:
                new_cost = cost_so_far + (1 if old_team != 2 else 0)
                if (ns1, ns2) not in new_dp or new_dp[(ns1, ns2)] > new_cost:
                    new_dp[(ns1, ns2)] = new_cost

            # Put person i in team3
            # effectively (s1, s2) doesn't change
            # but we must ensure s3 = prefix_sum[i+1] - (s1 + s2) isn't exceeding target:
            # s3 = prefix_sum[i+1] - s1 - s2
            # We'll check that it is <= target.
            s3 = prefix_sum[i+1] - s1 - s2
            if s3 <= target:
                ns1, ns2 = s1, s2
                new_cost = cost_so_far + (1 if old_team != 3 else 0)
                if (ns1, ns2) not in new_dp or new_dp[(ns1, ns2)] > new_cost:
                    new_dp[(ns1, ns2)] = new_cost

        dp = new_dp

    # Finally, we want s1=target, s2=target, and s3=target
    # That means check dp[(target, target)] if it exists.
    ans = dp.get((target, target), None)
    if ans is None:
        print(-1)
    else:
        print(ans)

# Do not forget to call main.
if __name__ == "__main__":
    main()
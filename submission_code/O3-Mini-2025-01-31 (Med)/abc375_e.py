def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    teams = []
    strengths = []
    total = 0
    for _ in range(N):
        a = int(next(it))
        b = int(next(it))
        teams.append(a)
        strengths.append(b)
        total += b

    # If the total strength is not divisible by 3, it’s impossible.
    if total % 3 != 0:
        sys.stdout.write("-1")
        return
    target = total // 3

    # Precompute prefix sums so that when processing first i persons, we know the total
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + strengths[i]

    # We now reformulate the problem:
    # We need to reassign each person i to a team in {1,2,3} (possibly staying on the same team)
    # so that the total strength in each team becomes exactly target.
    # Changing a person's team costs 1 move and leaving them in the original team costs 0.
    #
    # We set up a DP that processes persons one by one.
    # At step i (0-indexed), let s1 and s2 denote the accumulated strength in team1 and team2,
    # coming from the assignments of the first i persons.
    # The strength for team3 is then: prefix[i] - s1 - s2.
    # We must maintain that none of these partial totals exceed the target.
    # Our goal is to achieve s1 = target, s2 = target with team3 automatically reaching target 
    # (since total = 3 * target).
    #
    # dp[(s1, s2)] = minimal cost (number of moves) to reach a state with team1 having s1 and team2 having s2 
    # after assigning some persons.
    
    dp = {(0, 0): 0}  # initial state, no persons processed yet.
    
    for i in range(N):
        b = strengths[i]
        orig = teams[i]
        new_dp = {}
        current_total = prefix[i]
        new_total = prefix[i + 1]  # current_total + b
        for (s1, s2), cost in dp.items():
            # Choice 1: assign person i to team 1
            ns1 = s1 + b
            ns2 = s2
            t3 = new_total - ns1 - ns2  # team3's total after assignment
            if ns1 <= target and ns2 <= target and t3 <= target:
                new_cost = cost + (0 if orig == 1 else 1)
                key = (ns1, ns2)
                if key not in new_dp or new_dp[key] > new_cost:
                    new_dp[key] = new_cost

            # Choice 2: assign person i to team 2
            ns1 = s1
            ns2 = s2 + b
            t3 = new_total - ns1 - ns2
            if ns1 <= target and ns2 <= target and t3 <= target:
                new_cost = cost + (0 if orig == 2 else 1)
                key = (ns1, ns2)
                if key not in new_dp or new_dp[key] > new_cost:
                    new_dp[key] = new_cost

            # Choice 3: assign person i to team 3
            ns1 = s1
            ns2 = s2
            t3 = new_total - ns1 - ns2  # team3 gets b added in this option
            if ns1 <= target and ns2 <= target and t3 <= target:
                new_cost = cost + (0 if orig == 3 else 1)
                key = (ns1, ns2)
                if key not in new_dp or new_dp[key] > new_cost:
                    new_dp[key] = new_cost
        dp = new_dp

    ans = dp.get((target, target), None)
    sys.stdout.write(str(ans) if ans is not None else "-1")


if __name__ == '__main__':
    main()
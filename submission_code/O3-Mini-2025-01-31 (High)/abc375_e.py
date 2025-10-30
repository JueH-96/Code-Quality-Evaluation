def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Read input.
    N = int(data[0])
    idx = 1
    persons = []
    total = 0
    for i in range(N):
        a = int(data[idx])
        b = int(data[idx+1])
        idx += 2
        persons.append((a, b))
        total += b

    # If total strength is not divisible by 3, it's impossible.
    if total % 3 != 0:
        sys.stdout.write("-1")
        return
    T = total // 3
    
    # We use dynamic programming over persons.
    # Our dp state is a dictionary mapping (s1, s2) to the minimal number 
    # of switches needed so far, where s1 and s2 are the total strengths 
    # assigned to team 1 and team 2 respectively. The strength of team 3 
    # is implicitly (cumulative_strength - s1 - s2). In a valid state we need:
    #    s1 <= T, s2 <= T, and cumulative_strength - s1 - s2 <= T.
    #
    # Initially no persons have been processed.
    dp = {(0, 0): 0}
    cum = 0  # cumulative sum of strengths processed so far.
    
    # Process each person one by one.
    # For person i with original team A and strength b:
    # If we assign them to team j, we add b to that team's sum.
    # The cost addition is 0 if j equals the original team, 1 otherwise.
    # With team3, note that s1 and s2 remain unchanged (so team3 gets b).
    # But we must enforce that the new team3 sum (cum+b - s1 - s2) is ≤ T.
    #
    # Since every previous state is valid (s1+s2 >= cum - T),
    # for assignments to team1 or team2 the new s1+s2 increases by b so that
    # new state is automatically valid (team3 sum ≦ T). For team3 we explicitly
    # require that:
    #      s1+s2 >= (cum + b) - T.
    for (orig_team, b) in persons:
        new_cum = cum + b
        newdp = {}
        for (s1, s2), cost in dp.items():
            # Option 1: assign this person to team 1.
            ns1 = s1 + b
            ns2 = s2
            if ns1 <= T and ns2 <= T:
                # Since the sum s1+s2 increases by b, and we had
                # s1+s2 >= cum - T, we get ns1+ns2 >= new_cum - T automatically.
                new_cost = cost + (0 if orig_team == 1 else 1)
                st = (ns1, ns2)
                if st not in newdp or newdp[st] > new_cost:
                    newdp[st] = new_cost
            # Option 2: assign this person to team 2.
            ns1 = s1
            ns2 = s2 + b
            if ns1 <= T and ns2 <= T:
                new_cost = cost + (0 if orig_team == 2 else 1)
                st = (ns1, ns2)
                if st not in newdp or newdp[st] > new_cost:
                    newdp[st] = new_cost
            # Option 3: assign this person to team 3.
            # In this option the (s1,s2) state is unchanged, so team3’s strength becomes:
            # new_cum - (s1+s2) and we require that it does not exceed T.
            if (s1 + s2) >= (new_cum - T):  # equivalent to new team3 sum <= T
                new_cost = cost + (0 if orig_team == 3 else 1)
                st = (s1, s2)
                if st not in newdp or newdp[st] > new_cost:
                    newdp[st] = new_cost
        dp = newdp
        cum = new_cum

    # In the end, we want team1 and team2 to reach strength T.
    # Then team3 will automatically be total - T - T == T.
    ans = dp.get((T, T))
    if ans is None:
        sys.stdout.write("-1")
    else:
        sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()
def main():
    import sys
    input = sys.stdin.readline
    # Read input parameters
    N, X, Y = map(int, input().split())
    dishes = [tuple(map(int, input().split())) for _ in range(N)]
    
    # We want to maximize number of dishes subject to total sweetness <= X and total saltiness <= Y.
    # We maintain a list of states: each state is ((sweet, salt), count)
    # State ((s, t), c) means that we can achieve total sweetness s, saltiness t using c dishes.
    states = [((0, 0), 0)]
    
    for a, b in dishes:
        new_states = states.copy()
        for (s, t), cnt in states:
            ns, nt = s + a, t + b
            if ns <= X and nt <= Y:
                new_states.append(((ns, nt), cnt + 1))
        
        # We now prune "dominated" states.
        # A state u = ((s1, t1), cnt1) dominates v = ((s2, t2), cnt2) if s1 <= s2, t1 <= t2 and cnt1 >= cnt2.
        # We sort states by sweetness then saltiness (both increasing). This helps to check for domination.
        new_states.sort(key=lambda st: (st[0][0], st[0][1]))
        pruned = []
        for st in new_states:
            s, t = st[0]
            cnt = st[1]
            dominated = False
            # Since the list is small it is acceptable to check each state in pruned.
            for ps, pt in ((s2, t2) for (s2, t2), _ in pruned):
                # Also need the count from pruned state: so iterate with pruned state details.
            # We'll use pruned as list of full states.
            for pst in pruned:
                (ps, pt), pcnt = pst
                if ps <= s and pt <= t and pcnt >= cnt:
                    dominated = True
                    break
            if not dominated:
                pruned.append(st)
        states = pruned

    answer = 0
    for st in states:
        answer = max(answer, st[1])
    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()
def main():
    import sys

    data = sys.stdin.read().strip().split()
    # Parse input
    N = int(data[0])
    M = int(data[1])
    idx = 2

    # Wheels:
    #  C[i] = cost to play wheel i
    #  P[i] = number of possible outcomes for wheel i
    #  S[i] = list of outcomes
    C = [0]*N
    P = [0]*N
    S = [[] for _ in range(N)]

    for i in range(N):
        C[i] = float(data[idx]); idx += 1
        P[i] = int(data[idx]); idx += 1
        for _ in range(P[i]):
            S[i].append(int(data[idx]))
            idx += 1

    # If M=0 (not in this problem's constraints, but just in case), cost is 0:
    # but per constraints, M>=1 always.
    if M == 0:
        print(0.0)
        return

    # We model states 0..(M-1) as non-absorbing; state "M or more" is absorbing with cost 0.
    # For each wheel i, from each state x < M, we compute probability of going to each y < M
    # and the probability of going to the absorbing state >= M.

    # transitions[i][x][y] = Probability that using wheel i in state x (<M) goes to state y (<M).
    # trans_absorb[i][x]   = Probability that using wheel i in state x goes to >= M (absorbing).
    transitions = []
    trans_absorb = []
    for i in range(N):
        trans_i = []
        absorb_i = []
        for x in range(M):
            # Count how many outcomes drop us into each next state y < M or into absorbing
            count_next = [0]*(M)
            absorb_count = 0
            for val in S[i]:
                nx = x + val
                if nx >= M:
                    absorb_count += 1
                else:
                    count_next[nx] += 1
            # Convert to probabilities
            row_probs = [cnt/P[i] for cnt in count_next]
            absorb_prob = absorb_count/P[i]
            trans_i.append(row_probs)
            absorb_i.append(absorb_prob)
        transitions.append(trans_i)
        trans_absorb.append(absorb_i)

    # We will solve this via (controlled) Markov Decision Process policy iteration.
    # State space: x in [0..M-1].  M is absorbing (cost=0).
    # Actions (wheels): i in [0..N-1].
    # Because sum S[i] > 0 for each i, the chain is guaranteed to be absorbed in finite time.

    # Policy iteration steps:
    # 1) We keep a "policy" array pi[x], meaning which wheel is chosen in state x.
    # 2) Policy evaluation: solve for dp[x] = expected cost from x under that fixed policy.
    #    dp[x] = cost_of_wheel + sum_{y<M} T[x->y]*dp[y].  We do an iterative approach (Gauss-Seidel).
    # 3) Policy improvement: for each x, we see if there's an action i that lowers dp[x].
    #    dp_action[i] = C[i] + sum_{y<M} transitions[i][x][y]*dp[y].
    #    If dp_action[i] < dp[x] then we update pi[x] = i. Repeat until stable.

    # Initialize a trivial policy (e.g. always wheel 0)
    pi = [0]*(M)
    dp = [0.0]*(M)  # Will hold the cost-to-absorption from each state.

    def evaluate_policy():
        # Gauss-Seidel iteration to solve
        # dp[x] = C[ pi[x] ] + sum_{y} transitions[ pi[x] ][x][y] * dp[y], for x<M
        # We'll do up to some max iterations or until it converges.
        for _ in range(600):
            diff = 0.0
            for x in range(M):
                i = pi[x]
                # new_dp = cost + sum_{y<M} T[x->y]*dp[y]
                val = C[i]
                tr = transitions[i][x]
                # add transitions to non-absorbing
                # (absorbing leads to 0 additional cost, so we ignore it)
                s = 0.0
                for y in range(M):
                    s += tr[y]*dp[y]
                val += s
                old = dp[x]
                dp[x] = val
                d = abs(val - old)
                if d > diff:
                    diff = d
            if diff < 1e-12:
                break

    def improve_policy():
        # Try to see if any action i yields a smaller cost than dp[x].
        # If so, update pi[x].
        changed = False
        for x in range(M):
            old_val = dp[x]
            old_i = pi[x]
            best_i = old_i
            best_val = old_val
            # check all wheels
            for i in range(N):
                val = C[i]
                tr = transitions[i][x]
                s = 0.0
                for y in range(M):
                    s += tr[y]*dp[y]
                val += s
                if val < best_val - 1e-15:
                    best_val = val
                    best_i = i
            if best_i != old_i:
                pi[x] = best_i
                changed = True
        return changed

    # Policy Iteration:
    # We'll alternate evaluation/improvement until no improvement or we hit a max iteration.
    for _ in range(200):  # Usually converges quickly
        evaluate_policy()
        ch = improve_policy()
        if not ch:
            break  # No more improvement => optimal policy found

    # dp[0] is the expected cost from 0 points
    ans = dp[0]

    # Print result with sufficient precision
    # The problem states an absolute/relative error of <= 1e-5 is acceptable.
    print(f"{ans:.9f}")

# Remember to call main()
if __name__ == "__main__":
    main()
import sys

# Use sys.stdin.readline for faster input
input = sys.stdin.readline

def solve():
    line1 = input().split()
    N = int(line1[0])
    M = int(line1[1])

    costs = []
    P = []
    S = []

    for _ in range(N):
        line = input().split()
        c = int(line[0]) # Costs are integers
        p = int(line[1])
        s_values = list(map(int, line[2:]))
        costs.append(c)
        P.append(p)
        S.append(s_values)

    # E[s] will store the minimum expected cost to reach score >= M starting from score s.
    # We need E[0], E[1], ..., E[M-1].
    # E[s] = 0.0 for s >= M.
    # Use an array of size M for states 0 to M-1.
    E = [0.0] * M

    # Compute E[s] for s from M-1 down to 0
    for s in range(M - 1, -1, -1):
        min_expected_cost_s = float('inf')

        for i in range(N): # Iterate through each wheel
            cost_i = costs[i]
            num_outcomes_i = P[i]
            scores_i = S[i]

            prob_score_0 = 0.0
            sum_future_E_positive_gain = 0.0

            for score in scores_i:
                if score == 0:
                    prob_score_0 += 1.0 / num_outcomes_i
                else: # score > 0
                    next_score = s + score
                    
                    # The expected future cost from next_score is E[next_score] if next_score < M, and 0 if next_score >= M.
                    if next_score < M:
                         # E array stores values for states 0 to M-1.
                         # E[next_score] gives the expected cost from next_score.
                        sum_future_E_positive_gain += (1.0 / num_outcomes_i) * E[next_score]
                    else: # next_score >= M. The cost from here is 0.
                        pass # sum_future_E_positive_gain += (1.0 / num_outcomes_i) * 0.0


            # The formula derived is E_s = (C_i + sum_{j: S_{i,j}>0, s+S_{i,j} < M} (1/P_i) E_{s+S_{i,j}}) / (1 - p_{i,0})
            # prob_score_0 is p_{i,0}
            # sum_future_E_positive_gain is sum_{j: S_{i,j}>0, s+S_{i,j} < M} (1/P_i) E_{s+S_{i,j}}
            
            # The problem guarantees sum(S_i,j) > 0 for each wheel, so there's at least one S_i,j > 0.
            # This implies prob_score_0 < 1 for all wheels.
            # So the denominator (1.0 - prob_score_0) is always positive.
            # Using a small tolerance for floating point comparison might be needed in general,
            # but since prob_score_0 is calculated from counts / total, it should be exact or
            # close enough to avoid division by zero in this problem context.
            
            cost_option_i = (cost_i + sum_future_E_positive_gain) / (1.0 - prob_score_0)
            
            min_expected_cost_s = min(min_expected_cost_s, cost_option_i)

        E[s] = min_expected_cost_s

    # The answer is the minimum expected cost from score 0.
    # Print with sufficient precision.
    print("{:.15f}".format(E[0]))

solve()
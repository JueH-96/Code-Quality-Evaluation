def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    X = list(map(int, data[2:2+M]))
    A = list(map(int, data[2+M:2+2*M]))

    # 1) Check total stone count
    total_stones = sum(A)
    if total_stones != N:
        print(-1)
        return

    # Pair up positions and counts, then sort by position
    pairs = sorted(zip(X, A), key=lambda x: x[0])

    # 2) We must have a stone in cell 1 from the start (otherwise impossible to fill cell 1)
    if pairs[0][0] != 1:
        print(-1)
        return

    # 3) Build prefix sums S so that S[i] = A_1 + ... + A_i
    #    and array X' so that X'[i] = position of the i-th pair
    #    We'll check feasibility: for each j in [1..M-1], we need S[j] >= X'[j+1] - 1
    S = [0]*(M+1)
    Xp = [0]*(M+1)
    for i in range(M):
        Xp[i+1] = pairs[i][0]
        S[i+1] = S[i] + pairs[i][1]

    for i in range(1, M):
        # Check feasibility up to the next cell
        if S[i] < Xp[i+1] - 1:
            print(-1)
            return

    # 4) Compute the step-function sum of P(i) over i=1..N-1
    #    where P(i) = sum of stones in all cells <= i
    #    We extend X'[M+1] = N so we can handle the final gap
    Xp.append(N)  # now Xp[M+1] = N
    cost_p = 0
    # S[i] = sum of A_1..A_i, Xp[i] = position of i-th pair, i=1..M
    # For each segment j = 1..M:
    #   from i = Xp[j].. Xp[j+1]-1, P(i) = S[j].
    #   Number of terms = Xp[j+1] - Xp[j].
    #   Contribution = S[j]*(Xp[j+1]- Xp[j]).
    for j in range(1, M+1):
        cost_p += S[j] * (Xp[j+1] - Xp[j])  # Xp[j+1] exists because we appended N

    # 5) We subtract the sum of i from i=1..N-1, which is (N-1)*N//2
    cost_move = cost_p - (N-1)*N//2

    # If everything is feasible so far, cost_move is our answer.
    # One last check (though if sum(A)=N and the prefix checks passed,
    # we should be good). We ensure no contradiction at the very end:
    # S[M] = sum(A) = N, so that is consistent.

    # cost_move can be negative if something were off, but theoretically it shouldn't be
    # if all conditions are satisfied. Still, a final check for safety:
    if cost_move < 0:
        # Should not happen under correct feasibility checks, but just in case:
        print(-1)
    else:
        print(cost_move)

# Do not forget to call main
main()
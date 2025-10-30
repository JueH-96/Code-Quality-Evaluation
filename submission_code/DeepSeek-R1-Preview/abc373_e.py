def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N

    total = sum(A)
    R = K - total

    if N == 0:
        print()
        return

    # Sort the global list in descending order
    G = sorted(A, reverse=True)

    # Adjust M to be at most N-1 because each S_i has N-1 elements
    M_adj = min(M, N-1) if N > 1 else 0

    # Precompute sum_topM_global and sum_after
    if N == 0:
        sum_topM_global = 0
    else:
        sum_topM_global = sum(G[:M_adj])

    sum_after = G[M_adj] if M_adj < len(G) else 0

    result = []

    for a_i in A:
        if M_adj == 0:
            # If M is 0, everyone is elected
            result.append(0)
            continue

        # Check if a_i is in the top M_adj elements
        if a_i >= G[M_adj - 1]:
            sum_topM_i = sum_topM_global - a_i
            if M_adj < len(G):
                sum_topM_i += sum_after
        else:
            sum_topM_i = sum_topM_global

        # Compute the numerator
        numerator = sum_topM_i + R - M_adj * a_i

        if numerator <= 0:
            X_i = 0
        else:
            # Compute ceil(numerator / (M_adj + 1))
            X_i = (numerator + M_adj) // (M_adj + 1)

        # Check if X_i is feasible
        if X_i > R or X_i < 0:
            result.append(-1)
        else:
            result.append(X_i)

    # Handle the case where M is 0
    if M == 0:
        result = [0] * N

    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()
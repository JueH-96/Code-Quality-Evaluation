def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    pull_tabs = []
    regular = []
    can_openers = []

    for _ in range(N):
        T = int(input[idx])
        idx += 1
        X = int(input[idx])
        idx += 1
        if T == 0:
            pull_tabs.append(X)
        elif T == 1:
            regular.append(X)
        else:
            can_openers.append(X)

    # Sort and compute prefix sums for each category
    pull_tabs.sort(reverse=True)
    regular.sort(reverse=True)
    can_openers.sort(reverse=True)

    # Compute prefix sums for pull_tabs
    pull_prefix = [0]
    for x in pull_tabs:
        pull_prefix.append(pull_prefix[-1] + x)

    # Compute prefix sums for regular
    regular_prefix = [0]
    for x in regular:
        regular_prefix.append(regular_prefix[-1] + x)

    # Compute prefix sums for can_openers (sum of X_i)
    can_openers_prefix = [0]
    for x in can_openers:
        can_openers_prefix.append(can_openers_prefix[-1] + x)

    max_sum = 0

    # Iterate over possible K (number of can_openers to take)
    max_k = min(M, len(can_openers))
    for K in range(0, max_k + 1):
        if K > M:
            continue

        # Compute sum of top K can_openers
        if K == 0:
            S = 0
        else:
            S = can_openers_prefix[K]

        # Compute R_max: min(S, len(regular))
        R_max = min(S, len(regular))

        # Now, compute for R from 0 to R_max, but this is too slow
        # So we'll try all possible R up to R_max, but this is O(N^2)
        # Instead, we'll compute for R = min(S, len(regular), M-K)
        # And also R = min(S, len(regular), M-K - len(pull_tabs))
        # But to keep it manageable, perhaps find the R that allows P to be as large as possible

        T = M - K
        if T < 0:
            continue

        R = min(S, len(regular), T)
        P = T - R
        if P < 0:
            P = 0
        else:
            P = min(P, len(pull_tabs))

        current_sum = regular_prefix[R] + pull_prefix[P]
        if current_sum > max_sum:
            max_sum = current_sum

    print(max_sum)

if __name__ == '__main__':
    main()
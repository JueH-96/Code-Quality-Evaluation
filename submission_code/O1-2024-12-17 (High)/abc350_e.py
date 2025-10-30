def main():
    import sys
    sys.setrecursionlimit(10**7)

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = int(data[1])
    X = int(data[2])
    Y = int(data[3])

    # E(0) = 0 by definition (no cost needed if already 0)
    # For N > 0, we have the recurrence:
    #   E(N) = min(
    #       X + E( floor(N/A) ),
    #       6/5 * Y + (1/5)*( E(N//2) + E(N//3) + E(N//4) + E(N//5) + E(N//6) )
    #   )
    #
    # We will compute E(N) with a post-order DFS (iterative) + memo.

    memo = {0: 0.0}  # Base case
    stack = [N]
    # We'll use a simple set to track which states we've "expanded" so we only
    # push their children once.
    expanded = set([0])

    while stack:
        top = stack[-1]
        if top in memo:
            # Already have its value
            stack.pop()
            continue
        if top not in expanded:
            expanded.add(top)
            # Push children (unless top == 0 or 1, but we handle generically)
            if top > 1:
                cA = top // A
                c2 = top // 2
                c3 = top // 3
                c4 = top // 4
                c5 = top // 5
                c6 = top // 6
                # Push only those children not yet in memo
                # (They will eventually compute their values)
                if cA not in memo:
                    stack.append(cA)
                if c2 not in memo:
                    stack.append(c2)
                if c3 not in memo:
                    stack.append(c3)
                if c4 not in memo:
                    stack.append(c4)
                if c5 not in memo:
                    stack.append(c5)
                if c6 not in memo:
                    stack.append(c6)
        else:
            # All children should be in memo now; we can compute E(top)
            stack.pop()
            if top == 0:
                memo[top] = 0.0
            else:
                cA = top // A
                direct_cost = float(X) + memo[cA]

                c2 = top // 2
                c3 = top // 3
                c4 = top // 4
                c5 = top // 5
                c6 = top // 6
                # From our rearranged formula for the die option:
                #   E(N) = (6/5)*Y + (1/5)*(E(N//2)+E(N//3)+E(N//4)+E(N//5)+E(N//6))
                dice_cost = 1.2 * float(Y) + 0.2 * (
                    memo[c2] + memo[c3] + memo[c4] + memo[c5] + memo[c6]
                )

                memo[top] = min(direct_cost, dice_cost)

    ans = memo[N]
    # Print with enough precision (error <= 1e-6 allowed)
    print(f"{ans:.9f}")
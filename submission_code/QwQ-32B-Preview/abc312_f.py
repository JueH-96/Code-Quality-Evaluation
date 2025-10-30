import sys
import bisect

def main():
    import sys
    import bisect
    from itertools import accumulate

    # Read input
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    items = []
    idx = 2
    for _ in range(N):
        T = int(input[idx])
        X = int(input[idx+1])
        items.append((T, X))
        idx += 2

    # Categorize items
    pull_tab = []
    regular = []
    can_openers = []
    for T, X in items:
        if T == 0:
            pull_tab.append(X)
        elif T == 1:
            regular.append(X)
        elif T == 2:
            can_openers.append(X)

    # Sort and compute prefix sums
    pull_tab.sort(reverse=True)
    pull_tab_prefix = list(accumulate(pull_tab, initial=0))
    
    can_openers.sort(reverse=True)
    can_openers_prefix = list(accumulate(can_openers, initial=0))
    
    regular.sort(reverse=True)
    regular_prefix = list(accumulate(regular, initial=0))

    # Initialize max_happiness
    max_happiness = 0

    # Determine the maximum possible K
    max_K = min(len(regular), M)

    # Iterate over K from 0 to max_K
    for K in range(max_K + 1):
        # Find the smallest C such that can_openers_prefix[C] >= K
        C = bisect.bisect_left(can_openers_prefix, K)
        if C > len(can_openers):
            continue  # Not enough can openers
        # Calculate P
        P = M - C - K
        if P < 0:
            continue  # Not enough items left for pull-tab cans
        # Calculate total happiness
        pull_happiness = pull_tab_prefix[min(P, len(pull_tab_prefix)-1)]
        regular_happiness = regular_prefix[K]
        total_happiness = pull_happiness + regular_happiness
        # Update max_happiness
        if total_happiness > max_happiness:
            max_happiness = total_happiness

    print(max_happiness)

if __name__ == "__main__":
    main()
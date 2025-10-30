def main():
    import sys
    from collections import defaultdict

    data = sys.stdin.read().strip().split()
    N = int(data[0])

    # Dictionary to store the top two deliciousness values per flavor
    flavor_best = defaultdict(lambda: [0, 0])

    idx = 1
    for _ in range(N):
        f = int(data[idx])
        s = int(data[idx+1])
        idx += 2

        # Update the top two deliciousness values for flavor f
        if s > flavor_best[f][0]:
            flavor_best[f][1] = flavor_best[f][0]
            flavor_best[f][0] = s
        elif s > flavor_best[f][1]:
            flavor_best[f][1] = s

    # Calculate the maximum satisfaction for the same flavor case
    max_same_flavor = 0
    # Also store the single highest deliciousness per flavor
    singles = []
    for f, (highest, second_highest) in flavor_best.items():
        if second_highest > 0:
            # same flavor satisfaction: highest + second_highest/2
            satisfaction_same = highest + (second_highest // 2)
            max_same_flavor = max(max_same_flavor, satisfaction_same)
        singles.append(highest)

    # Sort the single-highest deliciousness values in descending order
    singles.sort(reverse=True)

    # Calculate the maximum satisfaction for the different flavor case
    max_diff_flavor = 0
    if len(singles) >= 2:
        # different flavor satisfaction: sum of top two distinct
        max_diff_flavor = singles[0] + singles[1]

    # The final answer is the maximum between same-flavor and different-flavor results
    print(max(max_same_flavor, max_diff_flavor))

# Do not forget to call main()!
main()
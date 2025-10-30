import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    same_flavor = defaultdict(list)

    for _ in range(n):
        f, s = map(int, sys.stdin.readline().split())
        same_flavor[f].append(s)
        same_flavor[f].sort(reverse=True)
        if len(same_flavor[f]) > 2:
            same_flavor[f].pop()

    # Compute the maximum satisfaction for same flavor
    same_max = 0
    for f in same_flavor:
        if len(same_flavor[f]) >= 2:
            s0 = same_flavor[f][0]
            s1 = same_flavor[f][1]
            current = s0 + s1 // 2
            if current > same_max:
                same_max = current

    # Compute the maximum s for each flavor
    max_per_flavor = {}
    for f in same_flavor:
        max_per_flavor[f] = same_flavor[f][0]

    # Compute the sum of top two different flavors
    sum_diff = 0
    values = list(max_per_flavor.values())
    if len(values) >= 2:
        values_sorted = sorted(values, reverse=True)
        sum_diff = values_sorted[0] + values_sorted[1]

    # Determine the overall maximum satisfaction
    result = max(same_max, sum_diff)
    print(result)

if __name__ == "__main__":
    main()
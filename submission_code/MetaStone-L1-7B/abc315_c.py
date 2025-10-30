import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    flavor_dict = defaultdict(list)

    for _ in range(n):
        f, s = map(int, sys.stdin.readline().split())
        flavor_dict[f].append(s)

    max_same = 0

    # Compute same flavor case
    for f in flavor_dict:
        s_list = flavor_dict[f]
        if len(s_list) >= 2:
            s_list_sorted = sorted(s_list, reverse=True)
            s = s_list_sorted[0]
            t = s_list_sorted[1]
            current = s + (t // 2)
            if current > max_same:
                max_same = current

    # Compute different flavor case
    max_per_flavor = {}
    for f in flavor_dict:
        max_per_flavor[f] = max(flavor_dict[f])

    max_diff = 0
    max_list = list(max_per_flavor.values())

    if len(max_list) >= 2:
        max_list_sorted = sorted(max_list, reverse=True)
        max_diff = max_list_sorted[0] + max_list_sorted[1]

    # Determine the overall maximum
    overall_max = max(max_same, max_diff)
    print(overall_max)

if __name__ == "__main__":
    main()
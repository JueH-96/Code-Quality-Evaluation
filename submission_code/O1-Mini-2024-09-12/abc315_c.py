import sys
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    cups = []
    for i in range(N):
        F = int(data[1 + 2*i])
        S = int(data[2 + 2*i])
        cups.append((F, S))
    # Sort cups descending by S
    cups_sorted = sorted(cups, key=lambda x: -x[1])
    # Two different flavors
    s1, f1 = cups_sorted[0][1], cups_sorted[0][0]
    satisfaction_diff = -1
    for i in range(1, N):
        if cups_sorted[i][0] != f1:
            satisfaction_diff = s1 + cups_sorted[i][1]
            break
    # Two same flavors
    flavor_dict = defaultdict(list)
    for f, s in cups_sorted:
        flavor_dict[f].append(s)
    satisfaction_same = -1
    for s_list in flavor_dict.values():
        if len(s_list) >=2:
            s = s_list[0] + s_list[1]//2
            if s > satisfaction_same:
                satisfaction_same = s
    # Compute the maximum
    max_satisfaction = max([satisfaction_diff, satisfaction_same])
    print(max_satisfaction)

if __name__ == "__main__":
    main()
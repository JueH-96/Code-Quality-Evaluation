import sys
from collections import defaultdict

def max_satisfaction():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    index = 1
    flavors = defaultdict(list)

    for _ in range(N):
        F = int(data[index])
        S = int(data[index + 1])
        flavors[F].append(S)
        index += 2

    max_satisfaction = 0

    # Check for different flavors
    all_deliciousness = []
    for flavor in flavors:
        all_deliciousness.extend(flavors[flavor])
    all_deliciousness.sort(reverse=True)
    if len(all_deliciousness) >= 2:
        max_satisfaction = all_deliciousness[0] + all_deliciousness[1]

    # Check for same flavors
    for flavor in flavors:
        if len(flavors[flavor]) >= 2:
            flavors[flavor].sort(reverse=True)
            s = flavors[flavor][0]
            t = flavors[flavor][1]
            satisfaction = s + t // 2
            if satisfaction > max_satisfaction:
                max_satisfaction = satisfaction

    print(max_satisfaction)

if __name__ == "__main__":
    max_satisfaction()
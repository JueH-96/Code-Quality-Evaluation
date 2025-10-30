# YOUR CODE HERE

import sys

def max_satisfaction():
    n = int(input())
    flavors = {}
    for _ in range(n):
        f, s = map(int, input().split())
        if f not in flavors:
            flavors[f] = []
        flavors[f].append(s)

    max_s = 0
    for f in flavors:
        flavors[f].sort(reverse=True)
        if len(flavors[f]) > 1:
            max_s = max(max_s, flavors[f][0] + flavors[f][1])
        if len(flavors[f]) > 2:
            max_s = max(max_s, flavors[f][0] + flavors[f][2] // 2)

    print(max_s)

max_satisfaction()
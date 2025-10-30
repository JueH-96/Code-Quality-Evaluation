# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    pos = defaultdict(list)
    for i in range(n):
        pos[a[i]].append(i)
    for i in range(n):
        if not pos[b[i]]:
            return 'No'
        if abs(pos[b[i]][0] - i) > k:
            return 'No'
        pos[b[i]].pop(0)
    return 'Yes'

T = int(input())
for _ in range(T):
    print(solve())
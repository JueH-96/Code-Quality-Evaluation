import sys
from collections import defaultdict

def longest_common_prefix(s1, s2):
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return i

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().split()
    S.sort()
    ans = 0
    for i in range(N-1):
        ans += longest_common_prefix(S[i], S[i+1])
    print(ans)

solve()
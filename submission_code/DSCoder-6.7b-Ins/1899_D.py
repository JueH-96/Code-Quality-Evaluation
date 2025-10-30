import sys
from collections import Counter

def read_input():
    t = int(sys.stdin.readline().strip())
    for _ in range(t):
        n = int(sys.stdin.readline().strip())
        a = list(map(int, sys.stdin.readline().strip().split()))
        yield (n, a)

def solve(n, a):
    a = [2**i for i in a]
    count = Counter(a)
    pairs = 0
    for k in count.keys():
        pairs += count[k]*(count[k]-1)//2
    return pairs

for n, a in read_input():
    print(solve(n, a))
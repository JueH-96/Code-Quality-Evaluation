import sys
import itertools

def solve():
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    R = list(map(int, input[2:2+N]))
    
    ranges = [range(1, r+1) for r in R]
    
    for seq in itertools.product(*ranges):
        if sum(seq) % K == 0:
            print(' '.join(map(str, seq)))

solve()
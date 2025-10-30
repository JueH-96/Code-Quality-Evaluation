import sys
import math
from itertools import combinations

def is_arithmetic(subseq):
    if len(subseq) <= 2:
        return True
    diff = subseq[1] - subseq[0]
    for i in range(1, len(subseq) - 1):
        if subseq[i + 1] - subseq[i] != diff:
            return False
    return True

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    MOD = 998244353
    counts = [0] * (N + 1)
    
    for k in range(1, N + 1):
        if k <= 2:
            counts[k] = math.comb(N, k)
        else:
            counts[k] = sum(1 for combo in combinations(range(N), k) if is_arithmetic([A[i] for i in combo]))
    
    print(' '.join(str(count % MOD) for count in counts[1:N+1]))

if __name__ == "__main__":
    main()
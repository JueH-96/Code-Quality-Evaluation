# YOUR CODE HERE
import sys
import sys
import sys
import sys
from sys import stdin
import sys
import sys
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353
    conditions = [[] for _ in range(N+1)]
    invalid = False
    for _ in range(M):
        L, R, X = map(int, sys.stdin.readline().split())
        if L == R and X == L:
            invalid = True
        if L <= X <= R:
            conditions[X].append( (L, R) )
    if invalid:
        print(0)
        return
    # Precompute factorial
    fact = [1]*(N+1)
    for i in range(1, N+1):
        fact[i] = fact[i-1]*i % MOD
    # Initialize blocks by merging overlapping intervals
    intervals = []
    for p in range(1, N+1):
        for (L, R) in conditions[p]:
            intervals.append( (L, R) )
    intervals.sort()
    blocks = []
    current_L, current_R = 1,1
    for (L, R) in intervals:
        if L > current_R:
            blocks.append( (current_L, current_R) )
            current_L, current_R = L, R
        else:
            current_R = max(current_R, R)
    blocks.append( (current_L, current_R) )
    # Now, for each block, compute the number of valid permutations
    # Using DP
    from collections import defaultdict
    dp_total = 1
    for (a, b) in blocks:
        length = b - a +1
        dp = [ [0]*(length+2) for _ in range(length+2) ]
        # f[i][j] where i <= j, 1-based
        for i in range(length+2):
            for j in range(length+2):
                dp[i][j] =0
        # Initialize empty intervals
        for i in range(length+2):
            dp[i][i-1] =1
        # Precompute allowed positions
        allowed = [[] for _ in range(length+1)]
        for p in range(a, b+1):
            local_p = p -a +1
            forbidden = False
            for (L, R) in conditions[p]:
                if L <=a and R >=b:
                    forbidden = True
                    break
            if not forbidden:
                allowed_p = p -a +1
                allowed[local_p].append(local_p)
        # Fill DP
        for size in range(1, length+1):
            for i in range(1, length - size +2):
                j = i + size -1
                total =0
                # Iterate over possible p
                for p in range(i, j+1):
                    # Check if p is allowed
                    global_p = a + p -1
                    is_allowed = True
                    for (L, R) in conditions[global_p]:
                        if L <=a and R >=b:
                            is_allowed = False
                            break
                    if is_allowed:
                        total = (total + dp[i][p-1] * dp[p+1][j]) % MOD
                dp[i][j] = total
        dp_total = dp_total * dp[1][length] % MOD
    # Now, handle the blocks not covered by any intervals
    covered = [False]*(N+2)
    for (a, b) in blocks:
        for p in range(a, b+1):
            covered[p] = True
    remaining = []
    start = None
    for p in range(1, N+1):
        if not covered[p]:
            if start is None:
                start = p
        else:
            if start is not None:
                remaining.append( (start, p-1) )
                start = None
    if start is not None:
        remaining.append( (start, N) )
    for (a, b) in remaining:
        length = b -a +1
        dp = [ [0]*(length+2) for _ in range(length+2) ]
        for i in range(length+2):
            for j in range(length+2):
                dp[i][j] =0
        for i in range(length+2):
            dp[i][i-1] =1
        # All positions are allowed
        for size in range(1, length+1):
            for i in range(1, length - size +2):
                j = i + size -1
                total =0
                for p in range(i, j+1):
                    total = (total + dp[i][p-1] * dp[p+1][j]) % MOD
                dp[i][j] = total
        dp_total = dp_total * dp[1][length] % MOD
    print(dp_total)
if __name__ == "__main__":
    main()
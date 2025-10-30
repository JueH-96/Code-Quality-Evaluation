import sys
input = sys.stdin.read

def solve():
    data = input().split()
    T = int(data[0])
    results = []
    
    index = 1
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        
        # The key observation is that all points can be colored black if and only if
        # gcd(N, K) = 1. This is because the points that can be symmetrically colored
        # form a cycle of length gcd(N, K). If gcd(N, K) = 1, the cycle covers all points.
        # Otherwise, it covers only a fraction of the points.
        
        from math import gcd
        if gcd(N, K) == 1:
            results.append("Yes")
        else:
            results.append("No")
    
    # Output all results
    sys.stdout.write("
".join(results) + "
")
# YOUR CODE HERE
from collections import defaultdict

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = data[1]
    S = data[2:N+2]
    
    def is_subsequence(s, t):
        it = iter(s)
        return all(c in it for c in t)
    
    count = 0
    for i in range(N):
        for j in range(N):
            if is_subsequence(S[i] + S[j], T):
                count += 1
                
    print(count)

solve()
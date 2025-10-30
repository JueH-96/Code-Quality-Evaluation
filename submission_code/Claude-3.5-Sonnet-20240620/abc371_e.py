# YOUR CODE HERE
from collections import defaultdict

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    result = 0
    last_occurrence = defaultdict(lambda: -1)
    
    for i in range(N):
        distinct = 0
        seen = set()
        
        for j in range(i, N):
            if A[j] not in seen:
                distinct += 1
                seen.add(A[j])
            
            result += distinct
            
            if last_occurrence[A[j]] >= i:
                distinct -= 1
            
            last_occurrence[A[j]] = j
    
    print(result)

solve()
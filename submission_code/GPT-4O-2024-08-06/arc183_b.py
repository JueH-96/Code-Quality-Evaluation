# YOUR CODE HERE
import sys
from collections import defaultdict, Counter

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        
        A = list(map(int, data[index:index + N]))
        index += N
        
        B = list(map(int, data[index:index + N]))
        index += N
        
        # Group indices by their modulo (K+1)
        groups_A = defaultdict(list)
        groups_B = defaultdict(list)
        
        for i in range(N):
            mod_index = i % (K + 1)
            groups_A[mod_index].append(A[i])
            groups_B[mod_index].append(B[i])
        
        possible = True
        for mod_index in groups_A:
            if Counter(groups_A[mod_index]) != Counter(groups_B[mod_index]):
                possible = False
                break
        
        if possible:
            results.append("Yes")
        else:
            results.append("No")
    
    sys.stdout.write("
".join(results) + "
")
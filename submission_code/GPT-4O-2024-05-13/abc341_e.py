# YOUR CODE HERE
def process_queries(N, Q, S, queries):
    S = list(S)
    results = []
    
    for query in queries:
        parts = query.split()
        q_type = int(parts[0])
        L = int(parts[1]) - 1
        R = int(parts[2]) - 1
        
        if q_type == 1:
            for i in range(L, R + 1):
                S[i] = '0' if S[i] == '1' else '1'
        elif q_type == 2:
            is_good = True
            for i in range(L, R):
                if S[i] == S[i + 1]:
                    is_good = False
                    break
            results.append("Yes" if is_good else "No")
    
    return results

import sys
input = sys.stdin.read
data = input().splitlines()

N, Q = map(int, data[0].split())
S = data[1]
queries = data[2:]

results = process_queries(N, Q, S, queries)
for result in results:
    print(result)
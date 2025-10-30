def find_minimum_operations(T, test_cases):
    results = []
    
    for case in test_cases:
        N, P = case
        visited = [False] * (N + 1)
        cycle_count = 0
        
        for i in range(1, N + 1):
            if not visited[i]:
                cycle_count += 1
                x = i
                while not visited[x]:
                    visited[x] = True
                    x = P[x - 1]
        
        results.append(cycle_count - 1)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
test_cases = []

for _ in range(T):
    N = int(data[index])
    P = list(map(int, data[index + 1:index + 1 + N]))
    test_cases.append((N, P))
    index += 1 + N

results = find_minimum_operations(T, test_cases)

for result in results:
    print(result)
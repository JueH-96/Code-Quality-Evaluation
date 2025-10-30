def can_make_identical(T, test_cases):
    results = []
    
    for case in test_cases:
        N, K, A, B = case
        if A == B:
            results.append("Yes")
            continue
        
        # Create a mapping of indices for each value in A
        from collections import defaultdict
        
        value_positions = defaultdict(list)
        for index, value in enumerate(A):
            value_positions[value].append(index)
        
        # Check if we can transform A into B
        possible = True
        for index in range(N):
            target_value = B[index]
            if target_value not in value_positions:
                possible = False
                break
            
            # Check if we can reach any of the positions of target_value from index
            reachable = False
            for pos in value_positions[target_value]:
                if abs(pos - index) <= K:
                    reachable = True
                    break
            
            if not reachable:
                possible = False
                break
        
        results.append("Yes" if possible else "No")
    
    return results

import sys
input = sys.stdin.read
data = input().splitlines()

T = int(data[0])
test_cases = []

line_index = 1
for _ in range(T):
    N, K = map(int, data[line_index].split())
    A = list(map(int, data[line_index + 1].split()))
    B = list(map(int, data[line_index + 2].split()))
    test_cases.append((N, K, A, B))
    line_index += 3

results = can_make_identical(T, test_cases)
print("
".join(results))
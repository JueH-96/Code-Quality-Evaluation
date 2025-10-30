def count_passed_pairs(N, T, S, positions):
    left_ants = []
    right_ants = []
    
    for i in range(N):
        if S[i] == '0':
            left_ants.append(positions[i])
        else:
            right_ants.append(positions[i])
    
    left_ants.sort()
    right_ants.sort()
    
    count = 0
    j = 0
    
    for left in left_ants:
        while j < len(right_ants) and right_ants[j] < left:
            j += 1
        count += len(right_ants) - j
    
    return count

import sys
input = sys.stdin.read
data = input().splitlines()

N, T = map(int, data[0].split())
S = data[1]
positions = list(map(int, data[2].split()))

result = count_passed_pairs(N, T, S, positions)
print(result)
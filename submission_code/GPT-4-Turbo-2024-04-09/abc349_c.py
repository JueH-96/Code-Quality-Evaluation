def is_airport_code(S, T):
    n = len(S)
    t1 = T[:2]
    t2 = T[:3]
    
    # Check for T[:3] as a subsequence
    i = 0
    for char in S:
        if i < 3 and char.upper() == t2[i]:
            i += 1
        if i == 3:
            return True
    
    # Check for T[:2] + 'X' as a subsequence
    if T[2] == 'X':
        i = 0
        for char in S:
            if i < 2 and char.upper() == t1[i]:
                i += 1
            if i == 2:
                return True
    
    return False

import sys
input = sys.stdin.read
data = input().split()
S = data[0]
T = data[1]

if is_airport_code(S, T):
    print("Yes")
else:
    print("No")
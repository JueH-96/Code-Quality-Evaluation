import sys
from collections import defaultdict

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = [0] * (N + 1)
B = [0] * (N + 1)
for i in range(1, N + 1):
    A[i] = int(data[index])
    B[i] = int(data[index + 1])
    index += 2

total_sum = sum(B[1:])  # Sum of all strengths

if total_sum % 3 != 0:
    print(-1)
else:
    target = total_sum // 3
    INF = N + 1  # A value larger than maximum possible changes
    
    # Initialize DP with a dictionary
    dp_current = {(0, 0): 0}  # (sum1, sum2): min changes
    
    # Iterate over each person
    for p in range(1, N + 1):
        dp_next = {}  # New dictionary for next state
        for (s1, s2), cost in dp_current.items():
            bp = B[p]  # Strength of current person
            ap = A[p]  # Initial team of current person
            
            # Assign to team 1
            new_s1_1 = s1 + bp
            new_s2_1 = s2
            new_cost_1 = cost + (0 if ap == 1 else 1)
            if (new_s1_1, new_s2_1) not in dp_next or new_cost_1 < dp_next[(new_s1_1, new_s2_1)]:
                dp_next[(new_s1_1, new_s2_1)] = new_cost_1
            
            # Assign to team 2
            new_s1_2 = s1
            new_s2_2 = s2 + bp
            new_cost_2 = cost + (0 if ap == 2 else 1)
            if (new_s1_2, new_s2_2) not in dp_next or new_cost_2 < dp_next[(new_s1_2, new_s2_2)]:
                dp_next[(new_s1_2, new_s2_2)] = new_cost_2
            
            # Assign to team 3
            new_s1_3 = s1
            new_s2_3 = s2
            new_cost_3 = cost + (0 if ap == 3 else 1)
            if (new_s1_3, new_s2_3) not in dp_next or new_cost_3 < dp_next[(new_s1_3, new_s2_3)]:
                dp_next[(new_s1_3, new_s2_3)] = new_cost_3
        
        # Update dp_current to dp_next for the next iteration
        dp_current = dp_next
    
    # Check if the target sum is achieved
    if (target, target) in dp_current and dp_current[(target, target)] < INF:
        print(dp_current[(target, target)])
    else:
        print(-1)
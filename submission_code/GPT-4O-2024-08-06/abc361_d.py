# YOUR CODE HERE
def min_operations_to_transform(N, S, T):
    # Count the number of W and B in S and T
    count_S_W = S.count('W')
    count_S_B = S.count('B')
    count_T_W = T.count('W')
    count_T_B = T.count('B')
    
    # If the counts of W and B do not match, it's impossible
    if count_S_W != count_T_W or count_S_B != count_T_B:
        return -1
    
    # Calculate mismatches
    w_to_b = 0
    b_to_w = 0
    
    for s, t in zip(S, T):
        if s == 'W' and t == 'B':
            w_to_b += 1
        elif s == 'B' and t == 'W':
            b_to_w += 1
    
    # Minimum operations needed is the maximum of the two mismatches
    return max(w_to_b, b_to_w)

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
S = data[1]
T = data[2]

# Output the result
print(min_operations_to_transform(N, S, T))
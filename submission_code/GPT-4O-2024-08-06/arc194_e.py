# YOUR CODE HERE
def can_transform(N, X, Y, S, T):
    # Count zeros and ones in S
    count_zeros_S = S.count('0')
    count_ones_S = S.count('1')
    
    # Count zeros and ones in T
    count_zeros_T = T.count('0')
    count_ones_T = T.count('1')
    
    # Check if the counts match
    if count_zeros_S == count_zeros_T and count_ones_S == count_ones_T:
        return "Yes"
    else:
        return "No"

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])
S = data[3]
T = data[4]

result = can_transform(N, X, Y, S, T)
print(result)
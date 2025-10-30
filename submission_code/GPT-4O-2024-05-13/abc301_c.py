# YOUR CODE HERE
def can_win(S, T):
    # Define the set of characters that @ can be replaced with
    replaceable_chars = set('atcoder')
    
    # Count the frequency of each character in S and T
    from collections import Counter
    count_S = Counter(S)
    count_T = Counter(T)
    
    # Remove the @ counts from both counters
    count_S_at = count_S.pop('@', 0)
    count_T_at = count_T.pop('@', 0)
    
    # Check if the remaining characters can be matched
    for char in set(count_S.keys()).union(set(count_T.keys())):
        if char in replaceable_chars:
            if count_S[char] > count_T[char]:
                count_T_at -= (count_S[char] - count_T[char])
            else:
                count_S_at -= (count_T[char] - count_S[char])
        else:
            if count_S[char] != count_T[char]:
                return "No"
    
    # Check if we have enough @ to cover the differences
    if count_S_at >= 0 and count_T_at >= 0:
        return "Yes"
    else:
        return "No"

import sys
input = sys.stdin.read
data = input().strip().split()
S = data[0]
T = data[1]
print(can_win(S, T))
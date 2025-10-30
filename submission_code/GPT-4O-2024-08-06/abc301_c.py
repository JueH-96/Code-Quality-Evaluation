# YOUR CODE HERE
def can_win(S, T):
    from collections import Counter
    
    allowed_chars = set('atcoder')
    
    # Count characters in S and T
    count_S = Counter(S)
    count_T = Counter(T)
    
    # Calculate the differences for non-@ characters
    diff_S = {}
    diff_T = {}
    
    for char in count_S:
        if char != '@':
            if char in count_T:
                if count_S[char] > count_T[char]:
                    diff_S[char] = count_S[char] - count_T[char]
            else:
                diff_S[char] = count_S[char]
    
    for char in count_T:
        if char != '@':
            if char in count_S:
                if count_T[char] > count_S[char]:
                    diff_T[char] = count_T[char] - count_S[char]
            else:
                diff_T[char] = count_T[char]
    
    # Calculate the total needed replacements
    needed_S = sum(diff_S[char] for char in diff_S if char in allowed_chars)
    needed_T = sum(diff_T[char] for char in diff_T if char in allowed_chars)
    
    # Check if we have enough @ to cover the needed replacements
    if needed_S <= count_S.get('@', 0) and needed_T <= count_T.get('@', 0):
        return "Yes"
    else:
        return "No"

import sys
input = sys.stdin.read
data = input().split()
S = data[0]
T = data[1]

print(can_win(S, T))
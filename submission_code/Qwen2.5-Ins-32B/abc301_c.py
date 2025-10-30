import sys

def can_win_with_cheating(s, t):
    # Count the occurrences of each character in both strings
    count_s = [0] * 26
    count_t = [0] * 26
    at_s = at_t = 0
    
    for char in s:
        if char == '@':
            at_s += 1
        else:
            count_s[ord(char) - ord('a')] += 1
    
    for char in t:
        if char == '@':
            at_t += 1
        else:
            count_t[ord(char) - ord('a')] += 1
    
    # Check if the counts of 'a', 't', 'c', 'o', 'd', 'e', 'r' are within the allowed range
    allowed_chars = "atcoder"
    for char in allowed_chars:
        if count_s[ord(char) - ord('a')] > count_t[ord(char) - ord('a')]:
            if at_t < count_s[ord(char) - ord('a')] - count_t[ord(char) - ord('a')]:
                return "No"
        else:
            if at_s < count_t[ord(char) - ord('a')] - count_s[ord(char) - ord('a')]:
                return "No"
    
    # Check if the remaining '@' characters can be balanced
    if abs(at_s - at_t) > 7:
        return "No"
    
    return "Yes"

# Read input from stdin
s = input().strip()
t = input().strip()

# Solve the problem and write the answer to stdout
print(can_win_with_cheating(s, t))
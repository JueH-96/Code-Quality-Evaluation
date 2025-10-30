# YOUR CODE HERE
K = int(input())
S = input().strip()
T = input().strip()

def can_transform_with_one_operation(s, t):
    # Case 1: Already same
    if s == t:
        return True
    
    len_s = len(s)
    len_t = len(t)
    
    # Case 2: Replace operation (same length, differ by exactly one character)
    if len_s == len_t:
        diff_count = 0
        for i in range(len_s):
            if s[i] != t[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count == 1
    
    # Case 3: Delete operation (s is one character longer than t)
    if len_s == len_t + 1:
        # Try removing each character from s and see if it matches t
        for i in range(len_s):
            if s[:i] + s[i+1:] == t:
                return True
        return False
    
    # Case 4: Insert operation (t is one character longer than s)
    if len_t == len_s + 1:
        # Try inserting each character of t into s at each position
        for i in range(len_t):
            if s == t[:i] + t[i+1:]:
                return True
        return False
    
    # If length difference is more than 1, impossible with one operation
    return False

if can_transform_with_one_operation(S, T):
    print("Yes")
else:
    print("No")
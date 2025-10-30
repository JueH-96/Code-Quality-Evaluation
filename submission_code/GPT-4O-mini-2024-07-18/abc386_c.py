def can_transform_with_k_operations(k, s, t):
    # If the strings are already equal
    if s == t:
        return "Yes"
    
    len_s = len(s)
    len_t = len(t)
    
    # If the length difference is more than 1, we cannot transform one into the other with just one operation
    if abs(len_s - len_t) > 1:
        return "No"
    
    # If lengths are the same, we can only replace one character
    if len_s == len_t:
        # Count the number of differing characters
        differences = sum(1 for a, b in zip(s, t) if a != b)
        return "Yes" if differences == 1 else "No"
    
    # If lengths differ by 1, we can either insert or delete
    if len_s + 1 == len_t:
        # Check if we can insert a character into s to make it t
        for i in range(len_t):
            if s == t[:i] + t[i+1:]:
                return "Yes"
    elif len_s == len_t + 1:
        # Check if we can delete a character from s to make it t
        for i in range(len_s):
            if t == s[:i] + s[i+1:]:
                return "Yes"
    
    return "No"

import sys

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    k = int(data[0])
    s = data[1]
    t = data[2]
    
    result = can_transform_with_k_operations(k, s, t)
    print(result)

if __name__ == "__main__":
    main()
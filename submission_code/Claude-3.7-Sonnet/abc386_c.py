def can_convert_with_at_most_k_operations(s, t, k):
    # Case 1: S and T are already identical
    if s == t:
        return True
    
    # Case 2: If k=0 and strings are different, it's impossible
    if k == 0:
        return False
    
    # Check lengths
    len_s, len_t = len(s), len(t)
    
    # If lengths differ by more than k, it's impossible
    if abs(len_s - len_t) > k:
        return False
    
    # Case 3: Insert a character (S is one character shorter than T)
    if len_s + 1 == len_t:
        i = j = 0
        while i < len_s and j < len_t:
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                j += 1
        return i == len_s  # All characters of S were matched
    
    # Case 4: Delete a character (S is one character longer than T)
    elif len_s == len_t + 1:
        i = j = 0
        while i < len_s and j < len_t:
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                i += 1
        return j == len_t  # All characters of T were matched
    
    # Case 5: Replace a character (S and T have the same length)
    else:  # len_s == len_t
        diff_count = sum(1 for i in range(len_s) if s[i] != t[i])
        return diff_count <= k

# Read input
k = int(input())
s = input().strip()
t = input().strip()

# Print the result
if can_convert_with_at_most_k_operations(s, t, k):
    print("Yes")
else:
    print("No")
def can_win(S, T):
    # Count occurrences of each character in S and T (excluding @)
    count_S = {}
    count_T = {}
    
    for char in S:
        if char != '@':
            count_S[char] = count_S.get(char, 0) + 1
    
    for char in T:
        if char != '@':
            count_T[char] = count_T.get(char, 0) + 1
    
    # Count @ in S and T
    at_S = S.count('@')
    at_T = T.count('@')
    
    # Characters that can replace @
    valid_replacements = set(['a', 't', 'c', 'o', 'd', 'e', 'r'])
    
    # Calculate how many @ we need to use from S and T
    used_at_S = 0
    used_at_T = 0
    
    for char in set(count_S.keys()).union(count_T.keys()):
        s_count = count_S.get(char, 0)
        t_count = count_T.get(char, 0)
        
        if s_count > t_count:  # More of char in S than in T
            diff = s_count - t_count
            if char not in valid_replacements:
                return "No"  # Can't use @ to offset a character not in valid_replacements
            used_at_T += diff
        elif s_count < t_count:  # More of char in T than in S
            diff = t_count - s_count
            if char not in valid_replacements:
                return "No"  # Can't use @ to offset a character not in valid_replacements
            used_at_S += diff
    
    # Check if we have enough @ to offset these differences
    if used_at_S > at_S or used_at_T > at_T:
        return "No"
    
    # Remaining @ in S and T
    remain_at_S = at_S - used_at_S
    remain_at_T = at_T - used_at_T
    
    # If the total remaining @ is odd, we can't make them match
    if (remain_at_S + remain_at_T) % 2 != 0:
        return "No"
    
    return "Yes"

# Read input and print output
S = input()
T = input()
print(can_win(S, T))
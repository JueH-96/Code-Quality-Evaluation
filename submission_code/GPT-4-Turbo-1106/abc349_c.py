def is_airport_code(s, t):
    # Check if T is a direct subsequence of S
    def is_subsequence(s, t):
        it = iter(s)
        return all(c in it for c in t)
    
    # Check if T can be formed by taking a subsequence of length 2 from S and appending 'X'
    def can_form_with_x(s, t):
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if s[i].upper() == t[0] and s[j].upper() == t[1]:
                    return True
        return False
    
    # Check if T is a subsequence of S
    if is_subsequence(s, t):
        return "Yes"
    # Check if T can be formed by taking a subsequence of length 2 from S and appending 'X'
    elif len(t) == 3 and t[2] == 'X' and can_form_with_x(s, t[:2]):
        return "Yes"
    else:
        return "No"

# Read input from stdin
s = input().strip()
t = input().strip()

# Output the result
print(is_airport_code(s, t))
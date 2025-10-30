def is_valid_parentheses(s):
    count = 0
    for c in s:
        if c == '(':
            count += 1
        else:
            count -= 1
        if count < 0:
            return False
    return count == 0

def reverse_parentheses(s):
    return ''.join(')' if c == '(' else '(' for c in reversed(s))

def find_all_valid_substrings(s):
    n = len(s)
    valid = set()
    
    # Find all valid substrings
    for i in range(n):
        for j in range(i+1, n+1):
            substring = s[i:j]
            if is_valid_parentheses(substring):
                valid.add((i, j))
    return valid

def solve():
    # Read input
    N = int(input())
    S = input().strip()
    
    # Use set to store all possible strings
    possible = {S}
    
    # Keep trying until no new strings can be formed
    while True:
        new_possible = set()
        
        # For each current string
        for curr in possible:
            # Find all valid substrings that can be reversed
            valid_substrings = find_all_valid_substrings(curr)
            
            # Try reversing each valid substring
            for start, end in valid_substrings:
                # Create new string by reversing the substring
                new_str = curr[:start] + reverse_parentheses(curr[start:end]) + curr[end:]
                new_possible.add(new_str)
        
        # If no new strings were formed, break
        if new_possible.issubset(possible):
            break
            
        # Add new strings to possible set
        possible.update(new_possible)
    
    # Return number of possible strings modulo 998244353
    return len(possible) % 998244353

print(solve())
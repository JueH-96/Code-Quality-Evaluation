def is_valid_substring(s, l, r):
    """Check if s[l:r+1] is a valid parenthesis sequence (0-indexed)"""
    count = 0
    for i in range(l, r+1):
        if s[i] == '(':
            count += 1
        else:
            count -= 1
        if count < 0:
            return False
    return count == 0

def reverse_substring(s, l, r):
    """Reverse and flip substring s[l:r+1] (0-indexed)"""
    result = list(s)
    for i in range(l, r+1):
        # s[i] becomes ')' if s[l+r-i] is '(', and '(' if s[l+r-i] is ')'
        if s[l+r-i] == '(':
            result[i] = ')'
        else:
            result[i] = '('
    return ''.join(result)

def count_reachable_strings(s):
    n = len(s)
    visited = set([s])
    queue = [s]
    
    while queue:
        current = queue.pop(0)
        
        # Try all possible substrings
        for l in range(n):
            for r in range(l, n):
                if is_valid_substring(current, l, r):
                    new_string = reverse_substring(current, l, r)
                    if new_string not in visited:
                        visited.add(new_string)
                        queue.append(new_string)
    
    return len(visited)

# Read input
n = int(input())
s = input().strip()

# Solve and print answer
print(count_reachable_strings(s))
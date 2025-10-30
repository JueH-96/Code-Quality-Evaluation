def is_valid_parenthesis(s):
    count = 0
    for c in s:
        if c == '(':
            count += 1
        else:
            count -= 1
        if count < 0:
            return False
    return count == 0

def reverse_parenthesis(s):
    result = []
    for c in s:
        if c == '(':
            result.append(')')
        else:
            result.append('(')
    return ''.join(result)

def get_all_reachable_strings(s):
    from collections import deque
    
    visited = set()
    queue = deque([s])
    visited.add(s)
    
    while queue:
        current = queue.popleft()
        
        # Try all possible valid substrings
        n = len(current)
        for i in range(n):
            for j in range(i + 2, n + 1, 2):  # Only even length substrings can be valid
                substring = current[i:j]
                if is_valid_parenthesis(substring):
                    # Reverse this substring
                    reversed_substring = reverse_parenthesis(substring)
                    new_string = current[:i] + reversed_substring + current[j:]
                    
                    if new_string not in visited:
                        visited.add(new_string)
                        queue.append(new_string)
    
    return len(visited)

# Read input
n = int(input())
s = input().strip()

# Find answer
result = get_all_reachable_strings(s)
print(result % 998244353)
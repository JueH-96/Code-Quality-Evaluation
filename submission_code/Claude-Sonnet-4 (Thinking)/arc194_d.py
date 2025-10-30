from collections import deque

def get_valid_substrings(s):
    n = len(s)
    valid_substrings = []
    
    for l in range(n):
        count = 0
        for r in range(l, n):
            if s[r] == '(':
                count += 1
            else:
                count -= 1
                if count < 0:
                    break
            if count == 0:
                valid_substrings.append((l, r))
    
    return valid_substrings

def apply_operation(s, l, r):
    s = list(s)
    original = s[:]
    
    for i in range(l, r+1):
        j = l + r - i
        if original[j] == '(':
            s[i] = ')'
        else:
            s[i] = '('
    
    return ''.join(s)

def solve(s):
    visited = set()
    queue = deque([s])
    visited.add(s)
    
    while queue:
        current = queue.popleft()
        
        # Get all valid substrings of the current string
        valid_substrings = get_valid_substrings(current)
        
        for l, r in valid_substrings:
            new_s = apply_operation(current, l, r)
            if new_s not in visited:
                visited.add(new_s)
                queue.append(new_s)
    
    return len(visited) % 998244353

# Read input
n = int(input())
s = input().strip()

# Find answer
answer = solve(s)
print(answer)
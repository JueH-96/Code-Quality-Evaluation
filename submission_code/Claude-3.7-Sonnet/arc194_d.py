from collections import deque

def all_valid_substrings(s):
    valid_substrings = []
    for i in range(len(s)):
        count = 0
        for j in range(i, len(s)):
            if s[j] == '(':
                count += 1
            else:
                count -= 1
            if count == 0:
                valid_substrings.append((i, j))
            if count < 0:
                break
    return valid_substrings

def reverse_substring(s, l, r):
    s_list = list(s)
    for i in range(l, r+1):
        j = l + r - i
        if s[j] == '(':
            s_list[i] = ')'
        else:
            s_list[i] = '('
    return ''.join(s_list)

def count_distinct_sequences(s):
    seen = set([s])
    queue = deque([s])
    
    # Store the valid substrings of each distinct string
    valid_substrings_cache = {}
    
    while queue:
        current_s = queue.popleft()
        
        if current_s not in valid_substrings_cache:
            valid_substrings_cache[current_s] = all_valid_substrings(current_s)
        
        valid_substrings = valid_substrings_cache[current_s]
        
        for l, r in valid_substrings:
            new_s = reverse_substring(current_s, l, r)
            if new_s not in seen:
                seen.add(new_s)
                queue.append(new_s)
    
    return len(seen) % 998244353

# Read input from stdin
n = int(input().strip())
s = input().strip()

# Print output
print(count_distinct_sequences(s))
def get_majority(s):
    # Get majority of three bits
    return '1' if s.count('1') > s.count('0') else '0'

def apply_operation(s):
    # Apply one operation to get string of length len(s)/3
    result = ''
    for i in range(0, len(s), 3):
        group = s[i:i+3]
        result += get_majority(group)
    return result

def get_final_value(s, N):
    # Apply operation N times to get final value
    current = s
    for _ in range(N):
        current = apply_operation(current)
    return current

def min_changes_needed(s, N, target, pos=0, memo=None):
    if memo is None:
        memo = {}
    
    if (pos, s) in memo:
        return memo[(pos, s)]
    
    # Base case: if we've applied all operations
    if pos == N:
        return 0 if s == target else float('inf')
    
    # Get the current result
    current_result = get_final_value(s, N-pos)
    
    # If we already get the target value, no changes needed
    if current_result == target:
        memo[(pos, s)] = 0
        return 0
    
    min_changes = float('inf')
    length = len(s)
    
    # Try changing each position
    for i in range(length):
        # Create new string with i-th bit flipped
        new_s = s[:i] + ('0' if s[i] == '1' else '1') + s[i+1:]
        # Get result after this change
        result = get_final_value(new_s, N-pos)
        
        if result == target:
            min_changes = min(min_changes, 1)
        else:
            # Recursively try more changes
            further_changes = min_changes_needed(new_s, N, target, pos, memo)
            if further_changes != float('inf'):
                min_changes = min(min_changes, 1 + further_changes)
    
    memo[(pos, s)] = min_changes
    return min_changes

# Read input
N = int(input())
A = input().strip()

# Get the current final value
current_final = get_final_value(A, N)

# If current final value is 1, we need to change it to 0
# If current final value is 0, we need to change it to 1
target = '0' if current_final == '1' else '1'

# Calculate minimum changes needed
result = min_changes_needed(A, N, target)
print(result)
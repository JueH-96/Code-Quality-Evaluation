def majority(a, b, c):
    """Return the majority value among three bits."""
    return 1 if a + b + c >= 2 else 0

def apply_operation(s):
    """Apply one operation to string s."""
    result = []
    for i in range(0, len(s), 3):
        result.append(majority(int(s[i]), int(s[i+1]), int(s[i+2])))
    return ''.join(map(str, result))

def get_final_value(s, n):
    """Apply operation n times to get final value."""
    for _ in range(n):
        s = apply_operation(s)
    return int(s[0])

def min_changes_recursive(s, n, target):
    """Find minimum changes needed to achieve target value."""
    if n == 0:
        # Base case: string of length 1
        current = int(s[0])
        return 0 if current == target else 1
    
    # For each group of 3, we need to determine what value it should produce
    min_changes = float('inf')
    
    # Try all possible combinations for the next level
    def try_combinations(idx, changes, next_level):
        if idx == len(s):
            # We've processed all groups, recurse to next level
            next_s = ''.join(map(str, next_level))
            total_changes = changes + min_changes_recursive(next_s, n-1, target)
            return total_changes
        
        if idx % 3 == 0:
            # Start of a new group
            group = s[idx:idx+3]
            current_majority = majority(int(group[0]), int(group[1]), int(group[2]))
            
            # Try both possible target values for this group (0 or 1)
            results = []
            for target_val in [0, 1]:
                # Calculate minimum changes needed to make this group have target_val as majority
                min_group_changes = 3  # worst case
                
                # Try all 8 possible combinations for the group
                for mask in range(8):
                    new_group = []
                    group_changes = 0
                    for i in range(3):
                        if mask & (1 << i):
                            new_group.append(1)
                            if group[i] == '0':
                                group_changes += 1
                        else:
                            new_group.append(0)
                            if group[i] == '1':
                                group_changes += 1
                    
                    if majority(new_group[0], new_group[1], new_group[2]) == target_val:
                        min_group_changes = min(min_group_changes, group_changes)
                
                # Recurse with this choice
                result = try_combinations(idx + 3, changes + min_group_changes, next_level + [target_val])
                results.append(result)
            
            return min(results)
    
    return try_combinations(0, 0, [])

# Read input
n = int(input())
s = input().strip()

# Get current final value
current_final = get_final_value(s, n)

# Find minimum changes to flip the final value
target_final = 1 - current_final
min_changes = min_changes_recursive(s, n, target_final)

print(min_changes)
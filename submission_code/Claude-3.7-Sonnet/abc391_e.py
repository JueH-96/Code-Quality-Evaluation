def compute_final_bit(binary_string, N):
    """Apply the majority operation N times to a binary string."""
    memo = {}
    
    def get_node_value(level, pos):
        key = (level, pos)
        if key in memo:
            return memo[key]
        
        # Base case: level 0 (original string)
        if level == 0:
            value = binary_string[pos]
            memo[key] = value
            return value
        
        # For higher levels
        child_pos = [3 * pos, 3 * pos + 1, 3 * pos + 2]
        
        count_ones = sum(1 for i in range(3) if get_node_value(level - 1, child_pos[i]) == '1')
        value = '1' if count_ones >= 2 else '0'
        
        memo[key] = value
        return value
    
    return get_node_value(N, 0)

def min_changes_to_flip(binary_string, N):
    """
    Calculate the minimum number of changes required to flip the final bit.
    
    Args:
        binary_string: The original binary string.
        N: The number of operations to apply.
    
    Returns:
        The minimum number of changes required.
    """
    memo = {}
    
    def get_min_cost(level, pos, target):
        key = (level, pos, target)
        if key in memo:
            return memo[key]
        
        # Base case: level 0 (original string)
        if level == 0:
            current_value = binary_string[pos]
            result = 0 if current_value == target else 1
            memo[key] = result
            return result
        
        # For higher levels
        child_pos = [3 * pos, 3 * pos + 1, 3 * pos + 2]
        
        min_cost = float('inf')
        
        # Enumerate all 8 possible configurations of the three children
        for config in ['000', '001', '010', '011', '100', '101', '110', '111']:
            majority = '1' if config.count('1') >= 2 else '0'
            
            # Only consider configurations that match the target
            if majority == target:
                cost = sum(get_min_cost(level - 1, child_pos[i], config[i]) for i in range(3))
                min_cost = min(min_cost, cost)
        
        memo[key] = min_cost
        return min_cost
    
    # Get the final bit
    final_bit = compute_final_bit(binary_string, N)
    
    # Calculate the minimum number of changes to flip the final bit
    flipped_bit = '1' if final_bit == '0' else '0'
    return get_min_cost(N, 0, flipped_bit)

# Main function
def solve():
    N = int(input().strip())
    A = input().strip()
    return min_changes_to_flip(A, N)

if __name__ == "__main__":
    print(solve())
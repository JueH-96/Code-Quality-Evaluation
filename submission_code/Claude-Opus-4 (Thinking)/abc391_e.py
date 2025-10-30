# YOUR CODE HERE
def solve(bits, start, length):
    if length == 1:
        current_value = int(bits[start])
        cost_to_0 = 0 if current_value == 0 else 1
        cost_to_1 = 0 if current_value == 1 else 1
        return current_value, cost_to_0, cost_to_1
    
    child_length = length // 3
    children_info = []
    
    for i in range(3):
        child_start = start + i * child_length
        value, cost_0, cost_1 = solve(bits, child_start, child_length)
        children_info.append((value, cost_0, cost_1))
    
    # Current value is the majority of children values
    current_value = 1 if sum(info[0] for info in children_info) >= 2 else 0
    
    # To make this node 0, we need at least 2 children to be 0
    cost_to_0 = float('inf')
    for mask in range(8):
        # Count how many children will be 0 with this mask
        zeros = sum(1 for i in range(3) if not ((mask >> i) & 1))
        if zeros >= 2:  # At least 2 children are 0
            cost = 0
            for i in range(3):
                if (mask >> i) & 1:  # Child i should be 1
                    cost += children_info[i][2]
                else:  # Child i should be 0
                    cost += children_info[i][1]
            cost_to_0 = min(cost_to_0, cost)
    
    # To make this node 1, we need at least 2 children to be 1
    cost_to_1 = float('inf')
    for mask in range(8):
        # Count how many children will be 1 with this mask
        ones = sum(1 for i in range(3) if (mask >> i) & 1)
        if ones >= 2:  # At least 2 children are 1
            cost = 0
            for i in range(3):
                if (mask >> i) & 1:  # Child i should be 1
                    cost += children_info[i][2]
                else:  # Child i should be 0
                    cost += children_info[i][1]
            cost_to_1 = min(cost_to_1, cost)
    
    return current_value, cost_to_0, cost_to_1

# Main code
n = int(input())
bits = input().strip()
length = 3 ** n

current_value, cost_to_0, cost_to_1 = solve(bits, 0, length)

# We want to flip the current value
if current_value == 0:
    print(cost_to_1)
else:
    print(cost_to_0)
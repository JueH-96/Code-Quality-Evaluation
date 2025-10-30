def calculate_cost(n):
    """
    Calculate the total cost of removing all integers not less than 2 from the blackboard.

    Args:
    n (int): The initial integer written on the blackboard.

    Returns:
    int: The total cost of removing all integers not less than 2.
    """
    cost = 0
    while n >= 2:
        # Calculate the cost of the current operation
        cost += n
        
        # If n is even, the two new integers will be n // 2 and n // 2
        if n % 2 == 0:
            n = n // 2
        # If n is odd, the two new integers will be (n - 1) // 2 and (n + 1) // 2
        else:
            n = (n - 1) // 2 + (n + 1) // 2
    return cost

# Read the input from stdin
n = int(input())

# Calculate and print the total cost
print(calculate_cost(n))
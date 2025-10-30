import math

def find_A(B):
    """
    Find a positive integer A such that A^A = B.
    
    Args:
    B (int): The input number.
    
    Returns:
    int: A if A^A = B, -1 otherwise.
    """
    # Start from 1 and increment A until A^A exceeds B
    A = 1
    while True:
        # Calculate A^A
        power = A ** A
        
        # If A^A equals B, return A
        if power == B:
            return A
        
        # If A^A exceeds B, break the loop
        if power > B:
            break
        
        # Increment A
        A += 1
    
    # If no A is found, return -1
    return -1

# Read the input from stdin
B = int(input())

# Find and print A
print(find_A(B))
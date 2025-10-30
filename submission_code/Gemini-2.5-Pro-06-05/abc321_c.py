# YOUR CODE HERE
import sys

# It is good practice to set a higher recursion limit for recursive solutions,
# although the default limit is sufficient for this problem's constraints (max depth 10).
sys.setrecursionlimit(10**6)

# This list will store all the generated 321-like numbers.
s321_numbers = []

def generate(num):
    """
    Recursively generates 321-like numbers. A number is 321-like if its digits
    are strictly decreasing from left to right (e.g., 321, 950).
    
    This function is called with a number `num` that is already 321-like.
    It adds `num` to the list and then tries to generate longer 321-like numbers
    by appending a new digit smaller than the last digit of `num`.
    """
    # Add the current valid 321-like number to the list.
    s321_numbers.append(num)
    
    # Get the last digit of the current number.
    last_digit = num % 10
    
    # Generate longer numbers by appending a new digit `i` that is smaller
    # than the last digit. This preserves the strictly decreasing property.
    for i in range(last_digit):
        new_num = num * 10 + i
        generate(new_num)

def solve():
    """
    Main function to solve the problem.
    It generates all 321-like numbers, sorts them, and finds the K-th one.
    """
    # All 321-like numbers are positive and can be formed by starting with a 
    # single-digit number (1-9) and recursively appending smaller digits.
    # We start a generation process from each of these single-digit numbers.
    for i in range(1, 10):
        generate(i)
    
    # The numbers are generated but not in ascending order. For example,
    # generate(2) and its descendants are processed before 10 is generated from 1.
    # Therefore, we must sort the list to find the K-th smallest number.
    s321_numbers.sort()
    
    # Read the integer K from standard input.
    K = int(input())
    
    # The problem uses 1-based indexing for K. The K-th smallest number is at
    # index K-1 in our 0-indexed sorted list. The problem guarantees that K
    # is a valid index.
    print(s321_numbers[K - 1])

# Run the solution.
solve()
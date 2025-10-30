import sys

def solve():
    N = int(sys.stdin.readline())

    # Special case: The 1st smallest good integer is 0.
    if N == 1:
        print(0)
        return

    # For N > 1, we are looking for the (N-1)-th positive good integer.
    # The sequence of positive good integers (2, 4, 6, 8, 20, 22, ...)
    # corresponds directly to the sequence of natural numbers (1, 2, 3, 4, 5, 6, ...)
    # if we represent the natural numbers in base 5 and then map each base-5 digit
    # (0, 1, 2, 3, 4) to an even decimal digit (0, 2, 4, 6, 8) respectively.
    # This mapping is simply: decimal_digit = 2 * base5_digit.
    
    num_for_base5_conversion = N - 1
    
    # This list will store the transformed digits, initially in reverse order
    # (from least significant to most significant base-5 digit).
    transformed_digits = []
    
    # Perform the base-5 conversion and digit transformation.
    # The loop continues as long as there's a part of the number left to convert.
    while num_for_base5_conversion > 0:
        # Get the current rightmost base-5 digit (remainder when divided by 5).
        remainder = num_for_base5_conversion % 5
        
        # Transform this base-5 digit to its corresponding even decimal digit.
        transformed_digit = 2 * remainder
        
        # Add the string representation of the transformed digit to our list.
        transformed_digits.append(str(transformed_digit))
        
        # Move to the next base-5 digit by performing integer division by 5.
        num_for_base5_conversion //= 5
        
    # The digits were collected in reverse order (e.g., for 24, we got ['4', '2']).
    # To form the final number string, we need to reverse the list and then join its elements.
    print("".join(reversed(transformed_digits)))

# Call the function to execute the solution
solve()
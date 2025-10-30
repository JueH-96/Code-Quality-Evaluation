# YOUR CODE HERE
def solve():
    N = int(input())

    def is_326_like(num_val):
        """
        Checks if num_val is a 326-like number.
        A 326-like number is a three-digit positive integer 
        where the product of the hundreds digit (H) and tens digit (T) 
        equals the ones digit (O).
        
        Example: 326 is 326-like because H=3, T=2, O=6, and 3 * 2 = 6.
        Example: 400 is 326-like because H=4, T=0, O=0, and 4 * 0 = 0.
        
        Constraints: N is between 100 and 919. The search for a 326-like
        number will not exceed 919 (the largest 326-like number).
        Thus, num_val will always be a three-digit number when this function
        is called in a context where it might return True.
        """
        
        # Extract hundreds digit (H)
        # For num_val = XYZ, H = X
        h = num_val // 100
        
        # Extract tens digit (T) and ones digit (O)
        # For num_val = XYZ, remainder_after_hundreds = YZ
        remainder_after_hundreds = num_val % 100
        # T = Y
        t = remainder_after_hundreds // 10
        # O = Z
        o = remainder_after_hundreds % 10
        
        return h * t == o

    current_num = N
    while True:
        if is_326_like(current_num):
            print(current_num)
            return # Exit the solve function, as the answer is found
        current_num += 1

# Call the main solving function when the script is executed.
# This ensures the program reads input, runs the algorithm, and prints output.
solve()
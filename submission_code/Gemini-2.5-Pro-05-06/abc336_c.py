import sys

def solve():
    N = int(sys.stdin.readline())

    K = N - 1

    base5_digits_int_list = [] # This will store integer digits (0-4) in LSB-first order
    
    if K == 0:
        base5_digits_int_list.append(0)
    else:
        temp_K = K
        while temp_K > 0:
            remainder = temp_K % 5
            base5_digits_int_list.append(remainder) # Store the integer digit
            temp_K //= 5
    
    # base5_digits_int_list is currently in LSB-first order.
    # For example, if K=7 (which is 12 in base 5), the list is [2, 1].
    # We need to iterate through it in reverse to process digits from MSB-first.

    ans_chars = [] # List to build the characters of the final result string
    
    # Iterate from MSB to LSB. Since base5_digits_int_list is LSB..MSB,
    # iterate it in reversed order.
    for digit_base5 in reversed(base5_digits_int_list): # digit_base5 is an int (0-4)
        # Map the base-5 digit to an even digit for the good integer:
        # 0 -> 0*2 = 0
        # 1 -> 1*2 = 2
        # 2 -> 2*2 = 4
        # 3 -> 3*2 = 6
        # 4 -> 4*2 = 8
        good_digit = digit_base5 * 2
        
        ans_chars.append(str(good_digit)) # Convert the good digit to string and append
        
    print("".join(ans_chars))

if __name__ == '__main__':
    solve()
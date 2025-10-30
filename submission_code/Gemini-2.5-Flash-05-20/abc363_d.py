import sys

def solve():
    N = int(sys.stdin.readline())

    # Handle the special case where N=1, as 0 is the first palindrome.
    if N == 1:
        print(0)
        return

    # Adjust N: We're looking for the (N-1)-th positive palindrome,
    # since 0 is the 1st overall palindrome.
    N -= 1 

    length = 1 # Start checking for palindromes of length 1 (e.g., 1, 2, ..., 9)

    while True:
        # Calculate the number of palindromes for the current 'length'.
        # The number of digits in the first half that determines the palindrome is (length + 1) // 2.
        # For a first half of 'k' digits, where the first digit isn't zero,
        # there are 9 * (10**(k-1)) such numbers.
        # (length - 1) // 2 correctly calculates (k-1) for both odd and even lengths:
        # e.g., L=1, (1-1)//2 = 0; L=2, (2-1)//2 = 0; L=3, (3-1)//2 = 1; L=4, (4-1)//2 = 1.
        num_pal_at_current_length = 9 * (10 ** ((length - 1) // 2))

        # Check if N falls within the current length group
        if N < num_pal_at_current_length:
            break # Found the correct length group
        
        # If not, subtract the count for this length and move to the next length group
        N -= num_pal_at_current_length
        length += 1
    
    # At this point:
    # 'length' is the number of digits of the N-th smallest palindrome (among positive ones).
    # 'N' is the 0-indexed offset within this 'length' group.
    # We subtract 1 from N because the target N is 1-indexed within its group,
    # and we need a 0-indexed offset to add to the base value.
    N -= 1 

    # Determine the number of digits in the "first half"
    half_len = (length + 1) // 2
    
    # Calculate the base value for the "first half" number.
    # E.g., for half_len=1, base is 1 (for 1-9).
    # E.g., for half_len=2, base is 10 (for 10-99).
    first_half_base_val = 10**(half_len - 1)
    
    # The actual numerical value of the first half of the palindrome
    first_half_num = first_half_base_val + N
    
    # Convert the first half number to a string
    s_first_half = str(first_half_num)
    
    # Construct the second part of the palindrome by reversing the first half
    if length % 2 == 1: # Odd length palindrome (e.g., 121, 12321)
        # The last digit of s_first_half is the middle digit.
        # The reversed part should exclude this middle digit.
        s_reversed_part = s_first_half[:-1][::-1]
    else: # Even length palindrome (e.g., 1221, 123321)
        # The reversed part is simply the full reversed s_first_half.
        s_reversed_part = s_first_half[::-1]
        
    # Concatenate to form the final palindrome string
    result_str = s_first_half + s_reversed_part
    
    # Print the result as an integer
    print(int(result_str))

# Call the solve function to run the program
solve()
import sys

def solve():
    N_orig = int(sys.stdin.readline())

    if N_orig == 1:
        print(0)
        return

    # N_orig is 1-indexed for all palindromes (0, 1, 2, ..., 9, 11, ...)
    # 0 is the 1st palindrome.
    # For N_orig > 1, we are looking for the (N_orig-1)-th non-zero palindrome.
    # Let N be 1-indexed for non-zero palindromes.
    N = N_orig - 1

    length = 1
    while True:
        # k is the length of the first half string for palindromes of `length`
        # k = ceil(length / 2)
        k = (length + 1) // 2
        
        # Number of palindromes of `length`:
        # The first half is a k-digit number. Its first digit can be 1-9 (9 choices).
        # The remaining k-1 digits can be 0-9 (10 choices each).
        # So, there are 9 * 10^(k-1) distinct first-half strings.
        count_at_this_length = 9 * (10**(k - 1))
        
        if N <= count_at_this_length:
            # The N-th non-zero palindrome has this `length`.
            # The value of `k` is correct for constructing this palindrome.
            break
        
        N -= count_at_this_length
        length += 1

    # We have found `length` and `k`.
    # N is 1-indexed relative to palindromes of this `length`.
    
    # Determine the first half string.
    # The smallest k-digit number (not starting with 0) is 10**(k-1).
    # (e.g., if k=1, 10^0=1; if k=2, 10^1=10)
    
    # `N-1` is the 0-indexed offset from this smallest k-digit number.
    val_offset = N - 1
    
    first_half_val = (10**(k - 1)) + val_offset
    first_half_str = str(first_half_val)
    
    # Construct the second half of the palindrome string
    if length % 2 == 1: # Odd length palindrome (e.g., ABCBA from first_half "ABC")
        # The part to reverse is the first_half_str excluding its last character.
        # e.g., if first_half_str is "123" (k=3), part_to_reverse is "12".
        part_to_reverse = first_half_str[:-1] 
    else: # Even length palindrome (e.g., ABCCBA from first_half "ABC")
        # The part to reverse is the entire first_half_str.
        # e.g., if first_half_str is "123" (k=3), part_to_reverse is "123".
        part_to_reverse = first_half_str
        
    second_half_str = part_to_reverse[::-1] # Reverse the string segment
    
    result_str = first_half_str + second_half_str
    print(result_str)

if __name__ == '__main__':
    solve()
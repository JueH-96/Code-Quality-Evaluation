# YOUR CODE HERE
import sys

def solve():
    """
    Reads an integer N from stdin and prints the N-th smallest palindrome number.
    """
    # Read the input N
    try:
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        return

    # The 1st palindrome (smallest) is 0.
    if N == 1:
        print(0)
        return

    # For N > 1, we are looking for the (N-1)-th positive palindrome.
    # We use 'n' as the 1-based index for positive palindromes.
    n = N - 1

    # We determine the length of the target palindrome by checking groups of
    # palindromes of increasing length.
    length = 1
    while True:
        # Calculate the number of palindromes for the current `length`.
        # This is determined by the number of possible "first halves".
        half_len = (length + 1) // 2
        
        # The first digit of a positive palindrome cannot be 0.
        # The number of ways to form the first half is 9 * 10^(half_len - 1).
        count = 9 * (10**(half_len - 1))

        # If our target index 'n' falls within the count for this length,
        # we have found the correct length group.
        if n <= count:
            # The first number that can be a "first half" for this half_len.
            # (e.g., for half_len=1, it's 1; for half_len=2, it's 10).
            base = 10**(half_len - 1)
            
            # The 'n'-th number in this sequence gives us our first half.
            first_half_num = base + n - 1
            
            first_half_str = str(first_half_num)
            
            # Construct the second half by reversing the appropriate part of the first.
            if length % 2 == 1:
                # Odd length: e.g., for "12321", first_half is "123", second is reverse of "12".
                second_half_str = first_half_str[:-1][::-1]
            else:
                # Even length: e.g., for "1221", first_half is "12", second is reverse of "12".
                second_half_str = first_half_str[::-1]
            
            # The full palindrome is the concatenation.
            print(first_half_str + second_half_str)
            
            # We've found the answer, so we can exit the loop.
            break
        
        # If 'n' is larger, our target palindrome is longer.
        # Subtract the count for this length and check the next length.
        n -= count
        length += 1

solve()
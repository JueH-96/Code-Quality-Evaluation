# YOUR CODE HERE
import sys

def solve():
    # Read the input integer N from standard input.
    N = int(sys.stdin.readline())

    # The 1st smallest palindrome number is 0.
    # Handle this base case separately.
    if N == 1:
        print(0)
        return

    # If N > 1, we are looking for the (N-1)-th positive palindrome number.
    # Let M be the 1-based index of the positive palindrome we are seeking.
    M = N - 1 

    # We need to determine the length 'k' of the M-th positive palindrome.
    # We start by checking palindromes of length k=1.
    k = 1 
    
    # Calculate count_k, the number of positive palindromes of length k.
    # The formula for the number of positive palindromes of length k is 9 * 10^floor((k-1)/2).
    # For k=1, floor((1-1)/2)=0, count_k = 9 * 10^0 = 9. (Palindromes 1-9)
    # For k=2, floor((2-1)/2)=0, count_k = 9 * 10^0 = 9. (Palindromes 11-99)
    # For k=3, floor((3-1)/2)=1, count_k = 9 * 10^1 = 90. (Palindromes 101-999)
    # For k=4, floor((4-1)/2)=1, count_k = 9 * 10^1 = 90. (Palindromes 1001-9999)
    # etc.
    # We use pow() function which handles large integers in Python.
    count_k = 9 * pow(10, (k - 1) // 2)
    
    # Iterate through possible lengths k. In each step, check if M is greater than
    # the count of palindromes of the current length k. If it is, subtract this count
    # from M and proceed to the next length k+1. Continue until M is less than or equal
    # to the count for the current length k.
    while M > count_k:
        M -= count_k
        k += 1
        # Update count_k for the new length k
        count_k = 9 * pow(10, (k - 1) // 2)

    # After the loop:
    # 'k' is the length of the M-th positive palindrome.
    # 'M' is the 1-based rank of the target palindrome among all palindromes of length k.

    # Now, we construct the M-th palindrome of length k.
    
    # Calculate the length of the first half of the palindrome string 's'.
    # If k is odd, the first half includes the middle digit.
    # The length is ceil(k/2), which can be computed using integer division as (k + 1) // 2.
    half_len = (k + 1) // 2
    
    # Calculate the base number for the first half. This is the smallest number 
    # that can form the first half for a palindrome of length k.
    # For example, if half_len=1 (k=1 or k=2), base=1. 
    # If half_len=2 (k=3 or k=4), base=10.
    # If half_len=3 (k=5 or k=6), base=100.
    # The general formula is 10^(half_len - 1).
    base = pow(10, half_len - 1)
    
    # The number represented by the first half string 's' is determined by the base 
    # number plus the rank offset (M-1). M is 1-based rank, so M-1 gives the 0-based offset.
    first_half_num = base + M - 1
    
    # Convert the first half number to its string representation.
    s_str = str(first_half_num)
    
    # Construct the full palindrome string based on the first half 's_str'.
    if k % 2 == 1: # If k has odd length
        # The second half of the palindrome is the reverse of the first half string, 
        # *excluding* its last character (which is the middle digit of the palindrome).
        # Example: k=3, first_half_num=12 -> s_str="12". Palindrome = "12" + reverse("1") = "121".
        palindrome_str = s_str + s_str[:-1][::-1]
    else: # If k has even length
        # The second half of the palindrome is simply the reverse of the entire first half string.
        # Example: k=4, first_half_num=12 -> s_str="12". Palindrome = "12" + reverse("12") = "1221".
        palindrome_str = s_str + s_str[::-1]
    
    # Print the resulting palindrome number string to standard output.
    print(palindrome_str)

# Execute the main logic contained within the solve function.
solve()
def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    N = int(sys.stdin.readline().strip())
    
    # If N <= 10, the N-th palindrome is simply N-1 (since 0 is the 1st, 1 is the 2nd, etc.)
    if N <= 10:
        print(N - 1)
        return
    
    # Subtract the 10 single-digit palindromes (0..9)
    N -= 10
    
    # Function to return how many palindromes of length L exist
    # For L = 1, we have 10 palindromes [0..9].
    # For L >= 2, the count = 9 * 10^((L-1)//2).
    def count_palindromes_of_length(L):
        if L == 1:
            return 10
        # 9 * 10^((L-1)//2)
        return 9 * pow(10, (L - 1) // 2)
    
    # Function to construct the palindrome:
    # prefix is an integer; L is the total length
    def construct_palindrome(prefix, L):
        s = str(prefix)
        if L % 2 == 0:
            # Even length: mirror the entire prefix
            return s + s[::-1]
        else:
            # Odd length: mirror all but the last digit of prefix
            return s + s[-2::-1]
    
    # Now find which length group N falls into
    length = 2
    while True:
        c = count_palindromes_of_length(length)
        if N <= c:
            # N-th palindrome within this length group
            # For length >= 2, half-length = (length+1)//2
            half_len = (length + 1) // 2
            # The first prefix in this length group is 10^(half_len-1) (except L=1, handled above)
            start_prefix = pow(10, half_len - 1)
            # The Nth palindrome in this group has prefix = start_prefix + (N-1)
            prefix_val = start_prefix + (N - 1)
            # Construct the full palindrome
            pal_str = construct_palindrome(prefix_val, length)
            print(pal_str)
            return
        else:
            N -= c
            length += 1

# Do not forget to call main
main()
# Read input N
N = int(input())

# Handle the special case N=1
if N == 1:
    print(0)
else:
    # For N > 1, we are looking for the (N-1)-th positive palindrome
    # The positive palindromes are 1, 2, ..., 9, 11, 22, ..., 99, 101, ...
    target_idx = N - 1

    # Find the length L of the palindrome
    L = 1
    cum_count = 0
    while True:
        # Calculate prefix length for length L palindromes
        # For a palindrome of length L, the first ceil(L/2) digits determine it.
        # prefix_len = ceil(L/2)
        prefix_len = (L + 1) // 2
        
        # Number of positive palindromes of length L
        # This is equal to the number of possible prefixes of length prefix_len
        # where the first digit is non-zero.
        # The possible prefixes are numbers from 10^(prefix_len - 1) to 10^prefix_len - 1.
        # The count is (10^prefix_len - 1) - 10^(prefix_len - 1) + 1 = 9 * 10^(prefix_len - 1).
        count_L = 9 * (10**(prefix_len - 1))

        if target_idx <= cum_count + count_L:
            # The target palindrome is within this length group L
            # target_idx_in_group is the 1-based index within the current length group
            target_idx_in_group = target_idx - cum_count
            break
        cum_count += count_L
        L += 1

    # Now we know the length L and the index target_idx_in_group (1-based) within this length group.
    # Calculate the prefix number
    # The length of the prefix is ceil(L/2)
    prefix_len = (L + 1) // 2 # Recalculate prefix_len for clarity

    # The first prefix number for palindromes of length L is 10^(prefix_len - 1).
    start_prefix = 10**(prefix_len - 1)
    
    # The target_idx_in_group-th prefix number is start_prefix + (target_idx_in_group - 1)
    prefix_num = start_prefix + target_idx_in_group - 1

    # Construct the palindrome string from the prefix number
    s = str(prefix_num)
    if L % 2 == 1: # Odd length palindrome
        # Palindrome is formed by prefix + reverse(prefix excluding last char)
        # E.g., L=3, prefix="12". Palindrome = "12" + reverse("1") = "12" + "1" = "121".
        # s[:-1] gives the prefix string excluding the last character.
        palindrome_str = s + s[:-1][::-1]
    else: # Even length palindrome
        # Palindrome is formed by prefix + reverse(prefix)
        # E.g., L=4, prefix="12". Palindrome = "12" + reverse("12") = "12" + "21" = "1221".
        palindrome_str = s + s[::-1]

    # Print the result
    print(palindrome_str)
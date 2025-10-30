def solve(N):
    if N == 1:
        return 0
    
    N -= 1  # Adjust for 0-indexed (since 0 is the 1st palindrome)
    
    # Count palindromes by length
    length = 1
    count = 9  # 1-digit palindromes: 1,2,3,4,5,6,7,8,9 (excluding 0)
    
    while N > count:
        N -= count
        length += 1
        if length % 2 == 0:  # even length
            count = 9 * (10 ** ((length - 2) // 2))
        else:  # odd length
            count = 9 * (10 ** ((length - 1) // 2))
    
    # Now we know the N-th palindrome has 'length' digits
    # and it's the N-th palindrome of this length
    
    if length % 2 == 0:  # even length
        half_len = length // 2
        # First half can be from 10^(half_len-1) to 10^half_len - 1
        first_half = 10**(half_len - 1) + N - 1
        first_half_str = str(first_half)
        palindrome = first_half_str + first_half_str[::-1]
    else:  # odd length
        half_len = (length + 1) // 2
        # First half can be from 10^(half_len-1) to 10^half_len - 1
        first_half = 10**(half_len - 1) + N - 1
        first_half_str = str(first_half)
        palindrome = first_half_str + first_half_str[-2::-1]
    
    return int(palindrome)

N = int(input())
print(solve(N))
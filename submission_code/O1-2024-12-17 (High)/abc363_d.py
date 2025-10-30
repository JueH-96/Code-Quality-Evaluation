def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    N = int(sys.stdin.readline().strip())
    
    # Counts how many palindromes of exactly d digits (including 0 when d=1).
    def count_pal(d):
        if d == 1:
            # 0..9  => 10 palindromes
            return 10
        if d % 2 == 0:
            # even d => 9 * 10^((d/2)-1)
            return 9 * (10 ** ((d // 2) - 1))
        else:
            # odd d => 9 * 10^((d-1)//2)
            return 9 * (10 ** ((d - 1) // 2))
    
    # Builds the full palindrome from the "first half".
    # even=True means mirror the entire string.
    # even=False means mirror all but the last character.
    def build_palindrome(half_str, even):
        if even:
            return int(half_str + half_str[::-1])
        else:
            return int(half_str + half_str[-2::-1])
    
    # Returns the offset-th palindrome of digit length d (0-based offset).
    def get_pal(d, offset):
        # Special case: d=1 => just offset itself (0..9).
        if d == 1:
            return offset
        
        if d % 2 == 0:
            # even number of digits
            half_start = 10 ** ((d // 2) - 1)
            half_num = half_start + offset
            return build_palindrome(str(half_num), True)
        else:
            # odd number of digits
            half_start = 10 ** ((d - 1) // 2)
            half_num = half_start + offset
            return build_palindrome(str(half_num), False)
    
    # Find which digit-length bucket N falls into, then construct that palindrome.
    prefix_sum = 0
    # d=1 up to ~60 is more than enough to cover N up to 10^18.
    for d in range(1, 70):
        c = count_pal(d)
        if prefix_sum + c >= N:
            offset = N - prefix_sum - 1
            ans = get_pal(d, offset)
            print(ans)
            return
        prefix_sum += c
    
    # Should never reach here if constraints are correct.
    # Just in case, raise an error or print something.
    print("ERROR")  # Fallback (not expected).

# Don't forget to call main at the end!
main()
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0
        # For each ending index j with nonzero last digit d:
        for j in range(n):
            d = int(s[j])
            # if last digit is 0, skip because the problem says non-zero last digit
            if d == 0:
                continue
            # We want to count the number of substrings ending at j that are divisible by d.
            # Let substring X = 10*q + d. Its divisibility by d is equivalent to:
            #   if d in {2,5}: automatically (since d divides 10)
            #   if d in {1,3,7,9}: q divisible by d.
            #   if d == 4: q even.
            #   if d == 6: q divisible by 3.
            #   if d == 8: q divisible by 4.
            #
            # We “build” q = s[i:j] by scanning backwards.
            running_q = 0  # we build q (as an integer), note: can be huge so we use modulus when needed.
            # For divisibility tests that need the entire number (for d in {1,3,7,9})
            # we perform mod arithmetic.
            # For those that need just parity or mod small numbers, we can update accordingly.
            count_for_j = 0
            # Loop backward for starting index i, substring q = s[i:j] (empty when i==j means q=0)
            # We iterate i from j down to 0.
            # We use a multiplier to add the new digit.
            mult = 1
            # Also, we keep a variable for the “digit‐sum” mod 3 (for d==6) and mod? and mod 4 (for d==8)
            # and parity (for d==4).
            mod_d = None  # for d in {1,3,7,9} we will use mod d.
            if d in (1,3,7,9):
                mod_d = 0
            mod3 = 0  # for d==6
            even_flag = True  # for d==4: q even if its last digit is even; we update as integer mod 2.
            mod4 = 0     # for d==8: we want q mod 4 == 0.
            
            # When i == j, q is empty which we treat as 0.
            def check_q(q_val, length):
                # q_val here is built from s[i:j] (with s[j] not included).
                # For the empty q (i.e. length == 0), interpret q = 0.
                qq = 0 if length == 0 else q_val
                if d in (2,5):
                    # always valid.
                    return True
                elif d in (1,3,7,9):
                    # need qq % d == 0.
                    return (qq % d) == 0
                elif d == 4:
                    # need q even.
                    return (qq % 2) == 0
                elif d == 6:
                    # need qq % 3 == 0.
                    return (qq % 3) == 0
                elif d == 8:
                    # need qq % 4 == 0.
                    return (qq % 4) == 0
                else:
                    return False
            
            # The substring of length 1: q is empty.
            if check_q(0, 0):
                count_for_j += 1
            
            # Now, for each starting index i from j-1 downto 0, update q = (s[i]*mult + running_q)
            # where running_q represents the value of s[i+1:j] (the part we have already processed)
            running_q = 0
            mult = 1
            # We will compute the needed mod values on the fly.
            running_mod_d = 0  # for d in {1,3,7,9} 
            running_mod3 = 0   # for d==6
            running_mod4 = 0   # for d==8
            running_even = None  # for d==4, we only need to know last digit parity.
            for i in range(j-1, -1, -1):
                digit = int(s[i])
                running_q = digit * mult + running_q
                mult *= 10
                # To avoid huge numbers, we update the mod values:
                if d in (1,3,7,9):
                    running_mod_d = (digit * (10 ** (j - i - 1)) + running_mod_d) % d
                    # But note: since j-i (the length) can be large, we instead build the value
                    # using the same recurrence as running_q but mod d.
                    # So we re‐compute:
                    # r = (digit*mult_mod + running_mod_d)
                    # We can update mult_mod separately.
                    # For simplicity (and since d < 10) we can recompute:
                    # (This method is not the most efficient but works since d is small.)
                    pass
                if d == 6:
                    running_mod3 = (digit + running_mod3) % 3
                if d == 8:
                    # For mod 4, we only need last two digits. We will recompute:
                    running_mod4 = (digit * (10 ** (j - i - 1)) + running_mod4) % 4
                if d == 4:
                    # For parity, it’s enough to check the last digit of q,
                    # but note that when building q backwards the last digit of q is s[j-1] when i==j-1,
                    # and then remains unchanged.
                    if i == j-1:
                        running_even = (digit % 2 == 0)
                    # For i < j-1, the parity does not change because the units place remains the one
                    # from s[j-1].
                
                # In our “quick‐and–dirty” simulation we recompute the full q for correctness.
                # (In Python with arbitrary–precision ints this is acceptable if the number of iterations
                # is not huge.)
                if check_q(running_q, j - i):
                    count_for_j += 1
            ans += count_for_j
        return ans

# For local testing:
if __name__ == '__main__':
    sol = Solution()
    print(sol.countSubstrings("12936"))    # Expected output 11
    print(sol.countSubstrings("5701283"))  # Expected output 18
    print(sol.countSubstrings("1010101010"))  # Expected output 25
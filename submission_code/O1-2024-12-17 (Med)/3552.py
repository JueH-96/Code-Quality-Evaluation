class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        """
        We want the largest n-digit palindrome (no leading zeros) that is divisible by k.
        Because n can be up to 10^5, we cannot simply iterate over all n-digit numbers.
        
        Key idea:
          1) An n-digit palindrome is determined by its "first half" (plus one middle digit if n is odd).
             Let M = (n + 1) // 2 be the length of this half.
             - If n is even, H[0..M-1] mirrored forms the palindrome of length n.
             - If n is odd, H[0..M-2] + [H[M-1]] + reverse(H[0..M-2]) forms the palindrome.
          2) We want the palindrome to be divisible by k (where 1 <= k <= 9).
             We can compute the remainder of the palindrome mod k by looking at
             each digit in the half and its position(s) in the full palindrome.
          3) A more direct method: Start with all 9's in the half (which gives
             a plainly largest string half), then see what the remainder R is.
             We must "reduce" the sum of digits, mod k, to fix that remainder so the
             final palindrome becomes 0 mod k.
          4) We set up a small DP (of size M x k) to find how much we reduce each half-digit
             from 9 (call that reduction d in [0..9]) so that the sum of (d * factor[i]) ≡ R (mod k).
             - factor[i] is the "contribution" of H[i] to the palindrome mod k.
             - If we reduce H[i] by d, the digit becomes 9 - d.
             - We pick the smallest d for each position from left to right to keep the overall number largest.
          5) We also ensure the first digit of the half (H[0]) is not zero, i.e. H[0] = 9 - d != 0,
             so d != 9 at i=0. This ensures no leading zero in the final palindrome.
        
        Steps in detail:
          - Compute M = (n+1)//2.
          - Precompute powers-of-10 mod k to build a factor array f of length M:
            f[i] = (10^(leftPos) + 10^(rightPos)) % k for the mirrored digits;
            if n is odd and i == M-1, that digit is the true middle, so f[i] = 10^(middlePos) % k alone.
          - The sum of all '9' digits in H has remainder R = (9 * sum(f[i])) % k.
          - We define a DP table dp with size (M+1) x k, and a parent pointer to reconstruct:
              dp[i][rem] = True  means we can reach remainder 'rem' by choosing reductions
                  for H[:i].
              We want dp[M][R] = True in the end.
          - Transitions:
              dp[0][0] = True
              For i in [0..M-1]:
                for rem in [0..k-1] where dp[i][rem] is True:
                  - if i==0, we skip d=9 (which would make H[0]=0)
                  - else d in [0..9]
                  newRem = (rem + d*f[i]) % k
                  if not dp[i+1][newRem]:
                      dp[i+1][newRem] = True
                      parent[i+1][newRem] = (rem, d)
          - Reconstruct from dp[M][R]: go backward to find how much we reduced each digit.
            Then H[i] = 9 - d.
          - Build the palindrome string from H.
        
        The DP has O(M * k * 10) complexity, which is about 10 * 9 * (n/2) = 45 * (n/2) ~ 22.5n.
        This is borderline but can pass in optimized Python if done carefully.
        We then output the constructed palindrome.
        """

        # 1) Handle small edge cases quickly if desired,
        #    but the general DP handles n=1 etc. anyway.

        # Precompute M:
        M = (n + 1) // 2

        # Precompute powers of 10 mod k up to n (since k <= 9, this is not large).
        pow10 = [0] * (n+1)
        pow10[0] = 1 % k
        for i in range(1, n+1):
            pow10[i] = (pow10[i-1] * 10) % k

        # Build the factor array f[i], length= M
        # If n is even, each H[i] is mirrored twice, each digit contributes 10^(n-1-i) and 10^i
        # If n is odd, the last index M-1 is the middle digit, which contributes just one power of 10.
        f = [0]*M
        if n % 2 == 0:
            # n even
            # M = n//2
            for i in range(M):
                leftExp = n - 1 - i
                rightExp = i
                f[i] = (pow10[leftExp] + pow10[rightExp]) % k
        else:
            # n odd
            # M = (n+1)//2
            for i in range(M-1):
                leftExp = n - 1 - i
                rightExp = i
                f[i] = (pow10[leftExp] + pow10[rightExp]) % k
            # middle digit (only counted once)
            midExp = n//2  # the center index
            # exponent = n-1 - midExp
            f[M-1] = pow10[n-1 - midExp] % k

        # Compute the initial remainder R if all half digits = 9
        # That sum is 9 * sum(f)
        sumF = sum(f) % k
        R = (9 * sumF) % k

        # We'll do a DP over i in [0..M], remainder in [0..k-1].
        dp = [ [False]*k for _ in range(M+1) ]
        parent = [ [None]*k for _ in range(M+1) ]
        # dp[i][rem] = can we achieve remainder 'rem' using reductions for H[:i]?
        # parent[i][rem] = (prevRem, d) means we came from dp[i-1][prevRem] with
        #   newRem = (prevRem + d*f[i-1]) % k.

        dp[0][0] = True

        for i in range(M):
            fi = f[i]
            # If we're at the first digit of the half (i=0), we skip d=9 => H[0]=0 not allowed
            if i == 0:
                d_range = range(0, 9)  # 0..8
            else:
                d_range = range(0, 10)  # 0..9

            for rem in range(k):
                if not dp[i][rem]:
                    continue
                # We can transition from dp[i][rem] to dp[i+1][newRem]
                for d in d_range:
                    newRem = (rem + d*fi) % k
                    if not dp[i+1][newRem]:
                        dp[i+1][newRem] = True
                        parent[i+1][newRem] = (rem, d)

        # Now we want dp[M][R] to be True. By problem statement, a solution should exist.
        # Reconstruct the half H by going backward from (M,R).
        # d is how much we reduced from '9', i.e. H[i] = 9 - d.
        if not dp[M][R]:
            # The problem statement suggests there's always a solution,
            # but in case none, we could return something like "" or handle separately.
            return ""  # or raise an error

        H = [0]*M
        curRem = R
        i = M
        while i > 0:
            prevRem, d = parent[i][curRem]
            # digit is 9 - d
            H[i-1] = 9 - d
            curRem = prevRem
            i -= 1

        # Now build the palindrome from H
        if n % 2 == 0:
            # even length
            left_str = "".join(str(x) for x in H)
            right_str = left_str[::-1]
            return left_str + right_str
        else:
            # odd length
            # the last digit H[M-1] is the middle
            left_str = "".join(str(x) for x in H[:-1])
            mid_str = str(H[-1])
            right_str = left_str[::-1]
            return left_str + mid_str + right_str
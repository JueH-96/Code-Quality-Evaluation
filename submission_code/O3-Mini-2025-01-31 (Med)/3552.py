class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # We want the largest n-digit palindrome divisible by k.
        # Our idea is to start from the maximum n‐digit palindrome (all digits 9)
        # and “reduce” it minimally (while preserving its palindromic form)
        # so that it becomes divisible by k.
        #
        # Let the independent half of the palindrome be determined by L = ceil(n/2) digits.
        # The full palindrome is determined by these L digits:
        #   - If n is even: palindrome = first half + reversed(first half)
        #   - If n is odd: palindrome = first half (with last digit as middle) + reversed(first half without the middle)
        #
        # Let d[i] (0 <= i < L) be the digits of the half.
        # In our candidate maximum, all digits are 9.
        # We will “reduce” some of these digits by subtracting a small nonnegative number x[i],
        # so that the new digit is 9 - x[i]. (We must keep the first digit at least 1; so x[0] can be at most 8.)
        #
        # The numeric value contributed by d[i] appears twice except possibly the middle digit.
        # Let pos1 = i and pos2 = n-1-i.
        # Then the contribution of digit d[i] is:
        #   weight = (10^(n-1-i) + (if i != n-1-i then 10^i else 0))
        # (For the middle digit when i == n-1-i, it contributes only once.)
        #
        # The maximum candidate (all 9’s) has value:
        #   P_max = sum_{i=0}^{L-1} 9 * weight[i]   (with the appropriate weight for each independent digit)
        # We denote the remainder "r" = P_max mod k.
        # To have a divisible palindrome, we want to subtract a total amount (call it X) that 
        # is congruent to r mod k. But the subtraction comes from adjusting each digit:
        #   We subtract (9 - (9 - x[i])) * weight[i] = x[i] * weight[i] from that contribution.
        # So we need:
        #   sum_{i=0}^{L-1} (x[i]*weight[i]) ≡ r  (mod k)
        # with 0 <= x[i] <= 9 for i>0 and 0 <= x[0] <= 8 (since the first digit cannot become 0).
        #
        # And, to maximize the final palindrome, we want each d[i] = 9 - x[i] as high as possible,
        # i.e. we want each x[i] to be as small as possible (in order from most significant to least).
        #
        # We will use a dynamic programming solution on the independent half positions.
        # Let L = ceil(n/2). For each position i and remainder rem mod k (0 <= rem < k)
        # define dp[i][rem] = True if it is possible to choose x[i], x[i+1], ..., x[L-1] so that
        #   sum_{j=i}^{L-1} (x[j]*weight[j]) ≡ rem (mod k).
        #
        # We compute dp backwards:
        #   dp[L][rem] = True if rem == 0, else False.
        # For i from L-1 downto 0, for each rem:
        #   dp[i][rem] = True if exists a choice x in allowed range for position i such that
        #      dp[i+1][ (rem - (x*weight[i]) mod k) ] is True.
        #
        # Then we do a greedy reconstruction: starting with rem = "r" (that is, the amount we need to subtract),
        # and for i = 0 to L-1, choose the smallest x (so that d[i] = 9 - x is as high as possible) 
        # which allows a valid continuation.
        # Finally, construct the full palindrome from the half.
        
        # Compute L = ceil(n/2)
        L = (n + 1) // 2
        
        # Precompute the weights mod k for each independent position.
        # weight[i] = (10^(n-1-i) + (10^i if i != n-1-i else 0)) mod k.
        # For indices that coincide (middle digit in odd length), only one term.
        weights = [0] * L
        # Since k is small, we can use pow(10, exp, k)
        for i in range(L):
            # mirror index:
            mirror = n - 1 - i
            # if position i and its mirror are distinct:
            if i != mirror:
                weights[i] = (pow(10, n - 1 - i, k) + pow(10, i, k)) % k
            else:
                weights[i] = pow(10, i, k) % k
        
        # Compute candidate max remainder: all digits = 9
        candidate_mod = 0
        for i in range(L):
            candidate_mod = (candidate_mod + 9 * weights[i]) % k
        target = candidate_mod  # We need to subtract an amount which is congruent to candidate_mod mod k.
        
        # dp[i][rem] : possibility from position i (0-indexed in half) achieving remainder rem.
        # We have L positions (i from 0 to L-1), and dp[L] for base case.
        dp = [[False] * k for _ in range(L + 1)]
        dp[L][0] = True
        
        # We'll fill dp in reverse. For each position i, allowed subtraction x:
        # For i==0, x in [0,8] (since d[0]=9-x must be at least 1)
        # For other positions, x in [0,9].
        for i in range(L - 1, -1, -1):
            # Precompute allowed x range for position i.
            if i == 0:
                max_sub = 8  # because 9 - x >=1
            else:
                max_sub = 9
            # Precompute the possible subtraction contributions for all allowed x at position i.
            trans = [ (x * weights[i]) % k for x in range(max_sub + 1) ]
            for rem in range(k):
                possible = False
                for x in range(max_sub + 1):
                    # We want to check if from i+1 we can achieve remainder: (rem - trans[x]) mod k.
                    nxt = (rem - trans[x]) % k
                    if dp[i + 1][nxt]:
                        possible = True
                        break
                dp[i][rem] = possible
        
        # Now, we want to know if starting with remainder = target (the total amount we must subtract) is achievable.
        if not dp[0][target]:
            # According to problem statement, there will always be an answer.
            return ""
        
        # Greedily reconstruct the minimal x's from left to right.
        res_half_digits = [None] * L
        rem = target
        for i in range(L):
            if i == 0:
                max_sub = 8
            else:
                max_sub = 9
            trans = [ (x * weights[i]) % k for x in range(max_sub + 1) ]
            # Try x from smallest up so that d[i] = 9 - x is as high as possible.
            for x in range(max_sub + 1):
                nxt = (rem - trans[x]) % k
                if dp[i + 1][nxt]:
                    # choose this x, so final digit becomes 9 - x.
                    res_half_digits[i] = str(9 - x)
                    rem = nxt
                    break
        
        half = "".join(res_half_digits)
        # Now, reconstruct the full palindrome.
        if n % 2 == 0:
            # even: mirror the half completely
            full = half + half[::-1]
        else:
            # odd: last digit of half is the middle digit. Mirror the rest.
            full = half[:-1] + half[::-1]
        return full

# For example usage:
# sol = Solution()
# print(sol.largestPalindrome(3, 5))   # Output: "595"
# print(sol.largestPalindrome(1, 4))   # Output: "8"
# print(sol.largestPalindrome(5, 6))   # Output: "89898"
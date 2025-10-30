from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # Count of odds and evens in 1..n
        odd_total = (n + 1) // 2
        even_total = n // 2
        # Cap for counts
        KCAP = 10**15 + 5
        
        # dp[or][er][lp]: number of alternating ways with or odds left, er evens left,
        # and last_parity = lp (0=even,1=odd,2=start/no last)
        dp = [[[0]*3 for _ in range(even_total+1)] for __ in range(odd_total+1)]
        # Base case: no elements left => one empty completion
        for lp in range(3):
            dp[0][0][lp] = 1
        
        # Build dp by increasing total left
        for total in range(1, n+1):
            for or_ in range(0, odd_total+1):
                er = total - or_
                if er < 0 or er > even_total:
                    continue
                for lp in range(3):
                    ways = 0
                    # If last is even, must pick odd
                    if lp == 0 or lp == 2:
                        if or_ > 0:
                            ways += dp[or_-1][er][1]
                    # If last is odd, must pick even
                    if lp == 1 or lp == 2:
                        if er > 0:
                            ways += dp[or_][er-1][0]
                    # Cap
                    dp[or_][er][lp] = ways if ways <= KCAP else KCAP
        
        # If total permutations < k, return empty
        if dp[odd_total][even_total][2] < k:
            return []
        
        # Prepare list of remaining values
        rem = list(range(1, n+1))
        odd_rem, even_rem = odd_total, even_total
        last_parity = 2  # start
        ans = []
        
        for pos in range(n):
            # Try candidates in lex order
            for i, v in enumerate(rem):
                p = v & 1  # parity of v
                # Check if allowed
                if last_parity != 2 and p == last_parity:
                    continue
                # Count how many if we pick v
                if p == 1:
                    cnt = dp[odd_rem-1][even_rem][1]
                else:
                    cnt = dp[odd_rem][even_rem-1][0]
                if cnt >= k:
                    # choose v
                    ans.append(v)
                    rem.pop(i)
                    if p == 1:
                        odd_rem -= 1
                    else:
                        even_rem -= 1
                    last_parity = p
                    break
                else:
                    k -= cnt
            else:
                # no candidate found (shouldn't happen if counts correct)
                return []
        return ans
from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # Count of even and odd numbers in 1..n
        even_count = n // 2
        odd_count = n - even_count
        
        # DP[e][o][p]: number of alternating sequences using e evens and o odds,
        # where the next required parity is p (0=even, 1=odd)
        dp = [[[0, 0] for _ in range(odd_count + 1)] for _ in range(even_count + 1)]
        dp[0][0][0] = 1
        dp[0][0][1] = 1
        
        for e in range(even_count + 1):
            for o in range(odd_count + 1):
                if e == 0 and o == 0:
                    continue
                # Next pick even
                if e > 0:
                    dp[e][o][0] = e * dp[e - 1][o][1]
                # Next pick odd
                if o > 0:
                    dp[e][o][1] = o * dp[e][o - 1][0]
        
        # Prepare list of available numbers
        avail = list(range(1, n + 1))
        e_rem = even_count
        o_rem = odd_count
        result = []
        prev_parity = None  # None for first position
        
        # Build the k-th permutation by lex order
        for pos in range(n):
            found = False
            for idx, x in enumerate(avail):
                p = x & 1  # parity of x: 0=even,1=odd
                # Check alternating constraint
                if prev_parity is not None and p == prev_parity:
                    continue
                # Compute remaining counts if we pick x
                ne, no = e_rem, o_rem
                if p == 0:
                    ne -= 1
                else:
                    no -= 1
                # Next required parity
                next_p = 1 - p
                # Number of completions
                cnt = dp[ne][no][next_p]
                if cnt < k:
                    k -= cnt
                    continue
                # We choose x
                result.append(x)
                del avail[idx]
                e_rem, o_rem = ne, no
                prev_parity = p
                found = True
                break
            if not found:
                # k is out of range
                return []
        return result
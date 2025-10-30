class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = int(s, 2)
        max_c = len(s)
        op_count = self.precompute_op_count(max_c, k)
        
        @lru_cache(None)
        def dp(index, count, is_less):
            if index == len(s):
                if count == 0:
                    return 0  # number is 0, which is invalid
                if count == 1 and not is_less and int(s, 2) == 1:
                    return 1 if 0 <= k else 0
                if count == 1:
                    return 1 if 1 <= k else 0
                if op_count[count] + 1 <= k:
                    return 1
                else:
                    return 0
            total = 0
            if is_less:
                # Can set the bit to 0 or 1
                total += dp(index + 1, count, True)
                total += dp(index + 1, count + 1, True)
            else:
                bit = s[index]
                if bit == '1':
                    # Can set the bit to 0 or 1
                    total += dp(index + 1, count, True)
                    total += dp(index + 1, count + 1, s[index+1:] != '0'*(len(s)-index-1))
                else:
                    # Must set the bit to 0
                    total += dp(index + 1, count, s[index+1:] != '0'*(len(s)-index-1))
            return total % MOD
        
        if n == 0:
            return 0
        return dp(0, 0, False)
    
    def precompute_op_count(self, max_c, k):
        op_count = [0] * (max_c + 1)
        for c in range(1, max_c + 1):
            x = c
            count = 0
            while x > 1:
                x = bin(x).count('1')
                count += 1
                if count > k:
                    break
            op_count[c] = count
        return op_count
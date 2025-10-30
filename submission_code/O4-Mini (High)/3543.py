class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # prefix sums for zeros and ones
        count0 = [0] * (n + 1)
        count1 = [0] * (n + 1)
        for i, ch in enumerate(s):
            count0[i+1] = count0[i] + (1 if ch == '0' else 0)
            count1[i+1] = count1[i] + (1 if ch == '1' else 0)
        
        ans = 0
        # enumerate all substrings s[i:j+1]
        for i in range(n):
            for j in range(i, n):
                zeros = count0[j+1] - count0[i]
                ones  = count1[j+1] - count1[i]
                # check the k-constraint
                if zeros <= k or ones <= k:
                    ans += 1
        return ans
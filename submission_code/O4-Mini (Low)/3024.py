class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        mod = 10**9+7
        n = len(s)
        # 1) find all rotations j in [0,n) such that rotating s right by j gives t
        # build text = s+s[:-1], pattern = t
        text = s + s[:-1]
        pattern = t
        # build lps for pattern
        lps = [0]*n
        length = 0
        for i in range(1, n):
            while length > 0 and pattern[i] != pattern[length]:
                length = lps[length-1]
            if pattern[i] == pattern[length]:
                length += 1
            lps[i] = length
        # KMP search
        matches = []
        j = 0  # pattern index
        for i in range(len(text)):
            while j > 0 and text[i] != pattern[j]:
                j = lps[j-1]
            if text[i] == pattern[j]:
                j += 1
                if j == n:
                    # match at text pos i-n+1 => left rotation by i-n+1
                    start = i - n + 1
                    # convert to right rotation: j = (n - start) % n
                    rot = (n - start) % n
                    matches.append(rot)
                    j = lps[j-1]
        if not matches:
            return 0
        # count how many are zero vs non-zero
        cnt0 = matches.count(0)
        cnt1 = len(matches) - cnt0
        # compute powers
        m = n-1
        pow_m = pow(m, k, mod)
        neg1k = -1 if (k & 1) else 1
        inv_n = pow(n, mod-2, mod)
        # dp for sum==0 mod n after k ops
        # dp0 = ( (m^k) + (-1)^k * (n-1) ) / n
        dp0 = (pow_m + neg1k * (n-1)) % mod
        dp0 = dp0 * inv_n % mod
        # dp for sum==j!=0
        # dp1 = ( (m^k) + (-1)^(k+1) ) / n
        dp1 = (pow_m + (-neg1k)) % mod
        dp1 = dp1 * inv_n % mod
        # answer
        ans = (cnt0 * dp0 + cnt1 * dp1) % mod
        return ans
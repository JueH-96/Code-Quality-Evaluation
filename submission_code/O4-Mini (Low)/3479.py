class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        # positions of zeros with sentinels
        zero_pos = [0]
        for i, ch in enumerate(s, 1):
            if ch == '0':
                zero_pos.append(i)
        zero_pos.append(n+1)
        m = len(zero_pos) - 2  # actual zeros count

        # count substrings with zero zeros: segments of consecutive ones
        res = 0
        ones = 0
        for ch in s:
            if ch == '1':
                ones += 1
            else:
                res += ones * (ones + 1) // 2
                ones = 0
        res += ones * (ones + 1) // 2

        # for k zeros, k from 1 up to sqrt(n)
        import math
        max_k = int(math.isqrt(n))
        # prefix ones for computing ones in between
        # but we'll use length-k trick
        for k in range(1, max_k+1):
            # if k > m break
            if k > m: break
            kk = k * k
            # slide window over zero_pos indices 1..m
            # window: zeros at zero_pos[i]...zero_pos[j], j=i+k-1
            for i in range(1, m - k + 2):
                j = i + k - 1
                zi = zero_pos[i]
                zj = zero_pos[j]
                # extra ones available on left and right
                L = zi - zero_pos[i-1] - 1
                R = zero_pos[j+1] - zj - 1
                # ones inside minimal span
                base_ones = (zj - zi + 1) - k
                # needed extra ones x+y >= T
                T = kk - base_ones
                if T <= 0:
                    # all combinations of x in [0..L], y in [0..R]
                    res += (L+1) * (R+1)
                    continue
                if T > L + R:
                    continue
                # compute bad = # pairs with x+y < T
                # let a <= b
                a, b = L, R
                if a > b:
                    a, b = b, a
                # we count sums s=0..T-1
                t = T - 1
                # three cases
                if t <= a:
                    bad = (t+1) * (t+2) // 2
                elif t <= b:
                    # sum_{s=0..a}(s+1) + (t-a)*(a+1)
                    bad = (a+1)*(a+2)//2 + (t - a)*(a+1)
                else:
                    # t > b, but t <= a+b
                    # sum1 s=0..a: (a+1)*(a+2)/2
                    sum1 = (a+1)*(a+2)//2
                    # sum2 s=a+1..b: (b-a)*(a+1)
                    sum2 = (b - a)*(a+1)
                    # sum3 s=b+1..t of (a+b+1 - s)
                    cnt3 = t - b
                    # sum_{u=1..cnt3} (a+1 - u) = cnt3*(a+1) - cnt3*(cnt3+1)/2
                    sum3 = cnt3*(a+1) - (cnt3*(cnt3+1)//2)
                    bad = sum1 + sum2 + sum3
                total = (L+1)*(R+1)
                res += total - bad
        return res

# Example usage:
# sol = Solution()
# print(sol.numberOfSubstrings("00011"))  # Expected 5
# print(sol.numberOfSubstrings("101101"))  # Expected 16
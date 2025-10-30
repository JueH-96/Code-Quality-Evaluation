class Solution:
    def maximumXorProduct(self, a: int, b: int, n: int) -> int:
        MOD = 10**9 + 7
        # product function f(x) = (a^x)*(b^x)
        def f(x):
            return (a ^ x) * (b ^ x)
        
        # hill-climbing (steepest-ascent) for one starting x0
        def climb(x0):
            x = x0
            fx = f(x)
            # Try at most n iterations (each increases fx strictly)
            for _ in range(n):
                best_fx = fx
                best_i = -1
                # try flipping each bit
                for i in range(n):
                    xi = x ^ (1 << i)
                    fi = f(xi)
                    if fi > best_fx:
                        best_fx = fi
                        best_i = i
                if best_i < 0:
                    break
                # accept best flip
                x ^= (1 << best_i)
                fx = best_fx
            return fx
        
        # mask of n bits
        if n >= 60:
            mask = (1 << 60) - 1
        else:
            mask = (1 << n) - 1
        
        # prepare candidate seeds
        seeds = set()
        seeds.add(0)
        seeds.add(mask)
        seeds.add((~a) & mask)
        seeds.add((~b) & mask)
        seeds.add((~(a ^ b)) & mask)
        
        best = 0
        for x0 in seeds:
            # ensure x0 in [0, 2^n)
            x0 &= mask
            val = climb(x0)
            if val > best:
                best = val
        
        return best % MOD
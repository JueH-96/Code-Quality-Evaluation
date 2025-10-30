from typing import List
from collections import Counter, defaultdict

MOD = 10**9 + 7

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        # Precompute factorials and inverse factorials for n up to 1000
        maxn = n
        fact = [1] * (maxn + 1)
        inv = [1] * (maxn + 1)
        for i in range(1, maxn + 1):
            fact[i] = fact[i-1] * i % MOD
        inv[maxn] = pow(fact[maxn], MOD-2, MOD)
        for i in range(maxn, 0, -1):
            inv[i-1] = inv[i] * i % MOD
        
        def comb(a, b):
            if b < 0 or b > a:
                return 0
            return fact[a] * inv[b] % MOD * inv[a-b] % MOD
        
        # Build right frequency map initially (all except index 0 middle)
        Rfreq = Counter(nums)
        Lfreq = Counter()
        ans = 0
        
        # Iterate possible middle at j
        for j, x in enumerate(nums):
            # move the middle x from R to L
            Rfreq[x] -= 1
            if Rfreq[x] == 0:
                del Rfreq[x]
            
            # we need at least 2 on each side
            if j >= 2 and n - j - 1 >= 2:
                Lx = Lfreq.get(x, 0)
                Rx = Rfreq.get(x, 0)
                Ln = j - Lx
                Rn = (n - j - 1) - Rx
                
                # Precompute sum of C(freq,2) on R and on L for non-x uses
                sumC2_R = 0
                for w, cnt in Rfreq.items():
                    sumC2_R = (sumC2_R + comb(cnt, 2)) % MOD
                sumC2_L = 0
                for w, cnt in Lfreq.items():
                    sumC2_L = (sumC2_L + comb(cnt, 2)) % MOD
                
                # 1) Cases where x frequency v = 1+a+b >=3  <=> a+b>=2
                for a in range(3):      # picks of x on left
                    for b in range(3):  # picks of x on right
                        if a + b >= 2:
                            ways_left  = comb(Lx, a) * comb(Ln, 2-a) % MOD
                            ways_right = comb(Rx, b) * comb(Rn, 2-b) % MOD
                            ans = (ans + ways_left * ways_right) % MOD
                
                # 2) Cases where a+b == 1 => v = 2, we must enforce no other value gets freq 2
                # Case (a=1, b=0)
                if Lx >= 1:
                    # choose 1 x on left in C(Lx,1) ways
                    cx = comb(Lx, 1)
                    # choose 1 non-x on left: any of the Lfreq except x
                    # and choose 2 non-x on right, all three non-x vals distinct
                    # sum over the chosen left non-x value v:
                    temp = 0
                    # pairs in R of 2 distinct non-x not equal to v:
                    # total non-x positions Rn, total pairs distinct = C(Rn,2) - sum over w!=x C(freq_R[w],2)
                    total_pairs_R_distinct = (comb(Rn, 2) - (sumC2_R - comb(Rfreq.get(x,0),2))) % MOD
                    # but for each v we must exclude pairs involving v, and also exclude pairs that pick same w twice
                    for v, fl in Lfreq.items():
                        if v == x: 
                            continue
                        # Number of R positions with v:
                        fr = Rfreq.get(v, 0)
                        # valid R pairs: from R positions not v => size (Rn - fr)
                        # pairs of non-v distinct among themselves:
                        #   C(Rn-fr,2) - sum_{w != x, w != v} C(freq_R[w],2)
                        pairs_without_v = comb(Rn - fr, 2)
                        # sumC2_R_without_v = sumC2_R - C(fr,2) - C(freq_R[x],2) + C(freq_R[x],2)
                        #                  = sumC2_R - C(fr,2)
                        sumC2_R_without_v = (sumC2_R - comb(fr,2)) % MOD
                        valid_pairs = (pairs_without_v - sumC2_R_without_v) % MOD
                        temp = (temp + fl * valid_pairs) % MOD
                    ans = (ans + cx * temp) % MOD
                
                # Case (a=0, b=1)
                if Rx >= 1:
                    cx = comb(Rx, 1)
                    total_pairs_L_distinct = (comb(Ln, 2) - (sumC2_L - comb(Lfreq.get(x,0),2))) % MOD
                    temp = 0
                    for v, fr in Rfreq.items():
                        if v == x:
                            continue
                        fl = Lfreq.get(v, 0)
                        # L positions not v => Ln - fl
                        pairs_without_v = comb(Ln - fl, 2)
                        sumC2_L_without_v = (sumC2_L - comb(fl,2)) % MOD
                        valid_pairs = (pairs_without_v - sumC2_L_without_v) % MOD
                        temp = (temp + fr * valid_pairs) % MOD
                    ans = (ans + cx * temp) % MOD
            
            # add this x to Lfreq for next iteration
            Lfreq[x] += 1
        
        return ans % MOD
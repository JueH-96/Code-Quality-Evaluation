class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        # Convert string to list of ints for faster access
        digits = [int(ch) for ch in s]
        
        # Answer
        ans = 0
        
        # For divisibility by 3
        pre3 = 0
        cnt3 = [1, 0, 0]   # count of prefix-sums mod3
        
        # For divisibility by 9
        pre9 = 0
        cnt9 = [1] + [0]*8  # count of prefix-sums mod9
        
        # For divisibility by 7
        pre7 = 0
        # Precompute pow10 mod7 for exponents 0..5 (cycle length 6)
        pow10mod7 = [1]*6
        for i in range(1, 6):
            pow10mod7[i] = (pow10mod7[i-1] * 10) % 7
        # Modular inverses mod7 via Fermat (7 is prime): a^(5) mod7 is a^{-1}
        invPow7 = [pow(pow10mod7[i], 5, 7) for i in range(6)]
        # counts7[a][r] = number of t seen so far with (t mod6)==a and pre7[t]==r
        counts7 = [[0]*7 for _ in range(6)]
        
        # Previous f_k values (f_k for index i-1)
        f2_prev = 0
        f3_prev = 0
        f4_prev = 0
        f7_prev = 0
        f9_prev = 0
        
        for i, d in enumerate(digits):
            # 1) Add to answer based on last digit d and previous f_k
            if d != 0:
                if d in (1, 2, 5):
                    # Always divisible by 1,2,5 respectively
                    ans += (i + 1)
                elif d == 3 or d == 6:
                    # d=3 or 6 -> k=3
                    ans += (f3_prev + 1)
                elif d == 4:
                    # d=4 -> k=2
                    ans += (f2_prev + 1)
                elif d == 7:
                    # d=7 -> k=7
                    ans += (f7_prev + 1)
                elif d == 8:
                    # d=8 -> k=4
                    ans += (f4_prev + 1)
                elif d == 9:
                    # d=9 -> k=9
                    ans += (f9_prev + 1)
            
            # 2) Update f2 for substrings ending at i divisible by 2
            if d % 2 == 0:
                f2_cur = i + 1
            else:
                f2_cur = 0
            
            # 3) Update prefix mod3 and f3 (substrings divisible by 3)
            pre3 = (pre3 + d) % 3
            f3_cur = cnt3[pre3]
            cnt3[pre3] += 1
            
            # 4) Update prefix mod9 and f9 (substrings divisible by 9)
            pre9 = (pre9 + d) % 9
            f9_cur = cnt9[pre9]
            cnt9[pre9] += 1
            
            # 5) Update f4 for substrings ending at i divisible by 4
            #    A number is divisible by 4 iff its last two digits form a number divisible by 4.
            if i == 0:
                # single digit
                f4_cur = 1 if (d % 4 == 0) else 0
            else:
                last_two = digits[i-1] * 10 + d
                f4_cur = (1 if (d % 4 == 0) else 0) + (i if (last_two % 4 == 0) else 0)
            
            # 6) Update prefix mod7 and f7 (substrings divisible by 7)
            pre7 = (pre7 * 10 + d) % 7
            # Count substrings j..i with j>=1 via counts7, plus j=0 if pre7==0
            f7_cur = 1 if pre7 == 0 else 0
            i_mod = i % 6
            # Sum contributions from t = j-1 in [0..i-1]
            # For each residue class a = t mod6, need pre7[t] == pre7[i] * invPow7[(i - t)%6]
            for a in range(6):
                e = (i_mod - a) % 6
                need = (pre7 * invPow7[e]) % 7
                f7_cur += counts7[a][need]
            # Now include current prefix in counts7
            counts7[i_mod][pre7] += 1
            
            # 7) Save cur -> prev for next iteration
            f2_prev, f3_prev, f4_prev, f7_prev, f9_prev = (
                f2_cur, f3_cur, f4_cur, f7_cur, f9_cur
            )
        
        return ans
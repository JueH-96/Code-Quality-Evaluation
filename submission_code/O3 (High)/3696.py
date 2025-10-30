class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0

        # helper: char to int fast
        digits = [ord(c) - 48 for c in s]

        # -------------------- data for the easy groups -------------------- #
        # prefix modulo 3 and 9 (numeric value mod 3 / 9)
        pref3 = 0
        pref9 = 0
        cnt3 = [1, 0, 0]          # counts of prefix value mod 3
        cnt9 = [1] + [0] * 8      # counts of prefix value mod 9

        # -------------------- data for the 7-group ------------------------ #
        # 10^k  (mod 7) for k=0..5
        pow10_mod7 = [1]
        for _ in range(5):
            pow10_mod7.append((pow10_mod7[-1] * 10) % 7)

        # inverses of 10^k (mod 7);  7 is prime so x^(5) == x^{-1} (mod 7)
        inv10_mod7 = [pow(x, 5, 7) for x in pow10_mod7]

        pref7 = 0
        cnt7 = [[0] * 7 for _ in range(6)]   # cnt7[index_mod6][value_mod7]
        cnt7[0][0] = 1                       # empty prefix

        for i, d in enumerate(digits):
            # update prefixes
            pref3 = (pref3 * 10 + d) % 3
            pref9 = (pref9 * 10 + d) % 9
            pref7 = (pref7 * 10 + d) % 7

            # ----------------------------------------------------------------
            # 1) digits for which every substring is valid (last digit 1,2,5)
            if d in (1, 2, 5):
                ans += i + 1                    # all substrings ending here

            # ----------------------------------------------------------------
            # 2) last digit 4 : need last two digits divisible by 4
            if d == 4:
                ans += 1                        # the single digit '4'
                if i >= 1:
                    val2 = digits[i - 1] * 10 + 4
                    if val2 % 4 == 0:
                        ans += i                # all longer substrings

            # ----------------------------------------------------------------
            # 3) last digit 8 : need last three digits divisible by 8
            if d == 8:
                ans += 1                        # single digit '8'
                if i >= 1:
                    val2 = digits[i - 1] * 10 + 8
                    if val2 % 8 == 0:
                        ans += 1                # length-2 substring
                if i >= 2:
                    val3 = digits[i - 2] * 100 + digits[i - 1] * 10 + 8
                    if val3 % 8 == 0:
                        ans += 1                # length-3 substring
                        ans += i - 2            # all longer substrings

            # ----------------------------------------------------------------
            # 4) last digits 3 or 6 : number must be divisible by 3
            if d in (3, 6):
                ans += cnt3[pref3]

            # ----------------------------------------------------------------
            # 5) last digit 9 : number must be divisible by 9
            if d == 9:
                ans += cnt9[pref9]

            # ----------------------------------------------------------------
            # 6) last digit 7 : use modular arithmetic with cycle 6
            if d == 7:
                i1_mod6 = (i + 1) % 6
                add = 0
                for k in range(6):                     # k = start index mod 6
                    L_mod6 = (i1_mod6 - k) % 6         # length mod 6
                    need = (pref7 * inv10_mod7[L_mod6]) % 7
                    add += cnt7[k][need]
                ans += add

            # ----------------------------------------------------------------
            # update counters for next positions
            cnt3[pref3] += 1
            cnt9[pref9] += 1
            cnt7[(i + 1) % 6][pref7] += 1

        return ans
from collections import defaultdict

class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        # we ignore substrings whose last digit is '0'
        
        # Precompute prefix arrays.
        # P3 and P9: mod 3 and 9 (note 10 % 3 = 1, 10 % 9 = 1)
        P3 = [0]*(n+1)  # used for d in {'3'}
        P9 = [0]*(n+1)  # used for d in {'9'}
        # For d='4', use mod4 with 10 mod4 = 2
        P4 = [0]*(n+1)
        # For d='6': use R6 with base = 10 mod6 = 4.
        R6 = [0]*(n+1)
        # For d='7': we compute a usual prefix mod7 with 10 mod7 = 3,
        # and then define Q7[k] = P7[k] * inv(3^k, 7)  mod 7.
        P7 = [0]*(n+1)
        power7 = [1]*(n+1)
        Q7 = [0]*(n+1)
        # For d='8': use R8 with base = 10 mod8 = 2.
        R8 = [0]*(n+1)
        
        for i in range(n):
            dgt = int(s[i])
            # For mod 3 and 9, 10 ≡ 1 so addition is enough.
            P3[i+1] = (P3[i] + dgt) % 3
            P9[i+1] = (P9[i] + dgt) % 9
            # For mod 4: 10 % 4 = 2.
            P4[i+1] = (P4[i]*2 + dgt) % 4
            # For mod 6: 10 % 6 = 4.
            R6[i+1] = (R6[i]*4 + dgt) % 6
            # For mod 7: 10 % 7 = 3.
            P7[i+1] = (P7[i]*3 + dgt) % 7
            power7[i+1] = (power7[i]*3) % 7
            # For mod 8: 10 % 8 = 2.
            R8[i+1] = (R8[i]*2 + dgt) % 8
        
        # Precompute Q7: for k >= 0, Q7[k] = P7[k]*inv(power7[k]) mod 7.
        # For k = 0, Q7[0] = 0.
        Q7[0] = 0
        for k in range(1, n+1):
            Q7[k] = (P7[k] * pow(power7[k], -1, 7)) % 7

        ans = 0
        # For digits 3, 9, 7 we use a frequency dictionary.
        # We want that for a substring s[i: j+1] (ending at index j) with d in {'3','9','7'},
        # the condition becomes:
        #   for d in {'3','9'}:  P[end] == P[i]   (since 10 ≡ 1)
        #   for d == '7':         Q7[end] == Q7[i]
        # In our loop we let the prefix array "value" at index j+1 correspond to the substring
        # ending at j. We want to count starting indices i in [0, j] that satisfy that.
        freq3 = defaultdict(int)
        freq3[P3[0]] = 1  # for starting index 0
        freq9 = defaultdict(int)
        freq9[P9[0]] = 1
        freq7 = defaultdict(int)
        freq7[Q7[0]] = 1
        # For d=='6' we use a dictionary (we count only starting indices i in [0, j) – single-digit is counted separately)
        freq6 = defaultdict(int)
        
        # The mapping for d=='6': with 10 mod6 = 4, one may show that for a substring s[i...j] ending with '6' 
        # the condition becomes: (R6[i]*4) mod 6 == R6[j+1].
        # Because 4 is not invertible mod6, one must “pre‐invert” the mapping.
        # One may check:
        #    If R6[j+1] == 0 then valid starting remainders are 0 and 3.
        #    If R6[j+1] == 2 then valid starting remainders are 2 and 5.
        #    If R6[j+1] == 4 then valid starting remainders are 1 and 4.
        #
        # We will maintain freq6 so that at each index j the dictionary contains R6[i] for i in [0, j) only.
        
        for j in range(n):
            d = s[j]
            # (Ignore if d=='0')
            if d == '0':
                # do nothing.
                pass
            elif d in {'1', '2', '5'}:
                # all substrings ending at j are valid.
                ans += (j + 1)
            elif d == '3':
                # Using P3: s[i...j] mod 3 = (P3[j+1] - P3[i]) mod 3 = 0   <=>  P3[j+1] == P3[i]
                ans += freq3[P3[j+1]]
            elif d == '9':
                ans += freq9[P9[j+1]]
            elif d == '7':
                ans += freq7[Q7[j+1]]
            elif d == '4':
                # For d == '4': if substring length = 1, always valid.
                # For length >= 2, one may show that because 10^L ≡ 0 for L>=2 (mod4),
                # the value of s[i...j] (with j-i+1 >=2) modulo 4 equals P4[j+1] (which is independent of i).
                # So the substring is divisible if and only if P4[j+1] == 0.
                # If P4[j+1] == 0 then there are j many starting indices i with i < j.
                ans += 1 + (j if P4[j+1] == 0 else 0)
            elif d == '6':
                # Single-digit substring always valid.
                # For a multi-digit substring (i.e. when the starting index i is strictly less than j)
                # we want to count how many starting indices i (that we have processed so far) satisfy:
                #   (R6[i]*4) mod 6 == R6[j+1]
                # Our frequency dictionary freq6 contains R6[i] for i in [0, j).
                target = R6[j+1]
                if target == 0:
                    valid = [0, 3]
                elif target == 2:
                    valid = [2, 5]
                elif target == 4:
                    valid = [1, 4]
                else:
                    valid = []
                ans += 1 + sum(freq6[r] for r in valid)
            elif d == '8':
                # Case for d=='8'
                # Single-digit substring is valid.
                # For a substring of length exactly 2 (s[j-1:j+1]) to be divisible by 8
                # (where >1 digit—the two–digit number can have a leading zero),
                # one may check that the first digit must be in { '0','4','8'}.
                # For substrings of length >= 3, note that since 10^L ≡ 0 (mod 8) for L ≥ 3,
                # the entire number s[i...j] (with length ≥ 3) modulo 8 equals R8[j+1],
                # so the substring is divisible if and only if R8[j+1] == 0.
                contrib = 1
                if j >= 1 and s[j-1] in {'0', '4', '8'}:
                    contrib += 1
                if j >= 2 and R8[j+1] == 0:
                    contrib += (j - 1)
                ans += contrib

            # For the frequency–based cases we update our dictionaries.
            # For d in {'3','9','7'} we want to count starting indices i in [0, j] for a substring ending at j.
            # We update after processing j so that these dictionaries always contain P[?] values for i in [0, j].
            freq3[P3[j+1]] += 1
            freq9[P9[j+1]] += 1
            freq7[Q7[j+1]] += 1
            # For d == '6', we need the dictionary to hold values for starting indices i in [0, j).
            # So update it at the end of the iteration with the value corresponding to i = j.
            freq6[R6[j]] += 1

        return ans
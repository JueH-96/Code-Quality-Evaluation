class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        res = 0

        # For last digit d in {1,2,5}: 
        # 10 ≡ 0 (mod d) so the substring value equals its last digit.
        # Hence if s[j] == d, all (j+1) substrings ending there are valid.
        for d in ['1', '2', '5']:
            for j, ch in enumerate(s):
                if ch == d:
                    res += (j + 1)

        # For d = 3: since 10 % 3 == 1, our recurrence P[k+1] = (P[k]*10 + digit) mod 3 reduces
        # the condition to: P[j+1] ≡ P[i] (mod 3). We use a frequency table.
        mod = 3
        current = 0
        freq = [0] * mod
        freq[0] = 1  # P[0] = 0
        for j, ch in enumerate(s):
            current = (current * 10 + int(ch)) % mod
            if ch == '3':
                res += freq[current]
            freq[current] += 1

        # For d = 9 (again 10 % 9 == 1)
        mod = 9
        current = 0
        freq = [0] * mod
        freq[0] = 1
        for j, ch in enumerate(s):
            current = (current * 10 + int(ch)) % mod
            if ch == '9':
                res += freq[current]
            freq[current] += 1

        # For d = 4:
        # Note: 10 % 4 = 2 and 10^2 % 4 = 0.
        # A substring ending with '4' always qualifies in the 1‐digit case.
        # For length >= 2, the number equals prefix[j+1] (because prefix[i]*0) and must be 0 mod4.
        mod = 4
        current = 0
        for j, ch in enumerate(s):
            current = (current * 10 + int(ch)) % mod
            if ch == '4':
                if j == 0:
                    res += 1
                else:
                    # count the 1-length substring
                    res += 1
                    # if current (i.e. P[j+1]) is 0, every substring (of length>=2) ending here works;
                    # there are j choices of start (i = 0,..., j-1)
                    if current == 0:
                        res += j

        # For d = 6:
        # Here 10 % 6 = 4 and for any length, 10^(L) % 6 is always 4.
        # The condition becomes: (P[j+1] - 4*P[i]) ≡ 0 (mod 6).
        # We do a frequency–pass while “looking back” using a small lookup.
        mod = 6
        current = 0
        freq = [0] * mod
        freq[0] = 1
        # mapping: for desired P[j+1] equal to r, we need P[i] to be any x with 4*x %6 == r.
        mapping = {0: [0, 3], 1: [],
                   2: [2, 5],
                   3: [],
                   4: [1, 4],
                   5: []}
        for j, ch in enumerate(s):
            current = (current * 10 + int(ch)) % mod
            if ch == '6':
                add = 0
                for x in mapping.get(current, []):
                    add += freq[x]
                res += add
            freq[current] += 1

        # For d = 7:
        # 10 mod 7 = 3 and the powers 10^(k) mod7 cycle with period 6:
        #   power[1] = 3, power[2] = 2, power[3] = 6, power[4] = 4, power[5] = 5, power[6] = 1.
        # The condition is: P[j+1] ≡ P[i] * power[(j-i+1)] (mod 7).
        # Since 7 is prime and 10 is invertible mod7, we “bucket” the prefix indices by their index mod6.
        mod = 7
        current = 0
        # Precompute power7 for exponents 1..6.
        power = {}
        power[1] = 10 % mod  # 10 mod7 = 3
        for exp in range(2, 7):
            power[exp] = (power[exp - 1] * (10 % mod)) % mod
        # Precompute inverses for these powers mod7.
        invpower = {}
        def modinv(a, m=7):
            return pow(a, m - 2, m)
        for exp in range(1, 7):
            invpower[exp] = modinv(power[exp], mod)
        # Maintain 6 buckets (for indices i in the P array, where i mod 6 is the bucket)
        freq_buckets = [[0] * mod for _ in range(6)]
        freq_buckets[0][0] = 1  # P[0] = 0 is in bucket 0.
        # Here, for a substring s[i...j] ending at index j with last digit '7',
        # its length is L = j-i+1.
        # If i is in bucket b then note that (j-i+1) mod6 is constant (namely, (j - b + 1) mod6),
        # and the condition becomes: P[i] ≡ (P[j+1] * inv(power[exp])) mod7.
        for j, ch in enumerate(s):
            current = (current * 10 + int(ch)) % mod  # now current == P[j+1]
            if ch == '7':
                total_add = 0
                for b in range(6):
                    # For any i with bucket b, we have:
                    exp = ( (j - b + 1) % 6 )
                    exp = exp if exp != 0 else 6
                    req = (current * invpower[exp]) % mod
                    total_add += freq_buckets[b][req]
                res += total_add
            bucket_index = (j + 1) % 6
            freq_buckets[bucket_index][current] += 1

        # For d = 8:
        # Since 10^3 % 8 == 0, when the substring length L >= 3 the value equals P[j+1].
        # In particular, for s[j]=='8':
        #   – the 1-digit substring is always valid
        #   – the 2-digit substring (i = j-1) is valid if the two–digit number is divisible by 8
        #   – any substring of length >=3 ending at j is valid if P[j+1] == 0.
        mod = 8
        P_arr = [0] * (n + 1)
        P_arr[0] = 0
        for j, ch in enumerate(s):
            P_arr[j + 1] = (P_arr[j] * 10 + int(ch)) % mod
            if ch == '8':
                contrib = 1  # 1-digit substring always works.
                if j >= 1:
                    two_digit = int(s[j - 1]) * 10 + 8
                    if two_digit % 8 == 0:
                        contrib += 1
                if j >= 2 and P_arr[j + 1] == 0:
                    # For length >= 3, any starting index i in 0..j-2 works.
                    contrib += (j - 1)
                res += contrib

        return res
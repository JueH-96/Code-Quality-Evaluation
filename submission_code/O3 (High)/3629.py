MOD = 10 ** 9 + 7

class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        """
        Keeps track, for every letter, of how many occurrences there are (mod MOD).
        One step needs only 26 assignments, so 26 * t ≤ 2.6·10⁶ for the given
        limits – easily fast enough.
        """
        # current amount of every letter (index 0 -> 'a', …, 25 -> 'z')
        cnt = [0] * 26
        for ch in s:
            cnt[ord(ch) - 97] += 1
        for i in range(26):
            cnt[i] %= MOD               # keep everything mod-MOD

        cur_len = len(s) % MOD          # total length mod MOD

        for _ in range(t):
            z = cnt[25]                 # how many 'z' we have this round

            # update total length: every 'z' produces one extra character
            cur_len = (cur_len + z) % MOD

            nxt = [0] * 26
            # letters 'a' … 'y' simply shift one step forward
            for i in range(25):         # 0 … 24
                nxt[i + 1] = cnt[i]

            # letter 'z' → "a" + "b"
            nxt[0] = (nxt[0] + z) % MOD
            nxt[1] = (nxt[1] + z) % MOD

            cnt = nxt                   # move to the next step

        return cur_len
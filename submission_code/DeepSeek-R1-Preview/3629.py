class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        if t == 0:
            return len(s) % MOD
        n = len(s)
        max_size = t + 26
        events = [0] * max_size
        for c in s:
            delay = (25 - (ord(c) - ord('a'))) % 26
            if delay < t:
                events[delay] += 1
        for k in range(t):
            current = events[k]
            if current == 0:
                continue
            k25 = k + 25
            if k25 < max_size:
                events[k25] = (events[k25] + current) % MOD
            k26 = k + 26
            if k26 < max_size:
                events[k26] = (events[k26] + current) % MOD
        total_z = sum(events[:t]) % MOD
        return (n + total_z) % MOD
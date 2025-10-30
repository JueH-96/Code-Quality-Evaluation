class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        from math import factorial

        # Helper function for fast exponentiation
        def pow_mod(x, y):
            result = 1
            x = x % MOD
            while y > 0:
                if y % 2:
                    result = (result * x) % MOD
                x = (x * x) % MOD
                y //= 2
            return result

        # Find segments of non-sick children
        segments = []
        start = 0 if sick[0] != 0 else -1
        for i in range(1, len(sick)):
            if sick[i] - sick[i-1] > 1:
                if start == -1:
                    segments.append((sick[i-1] + 1, sick[i] - 1, 'one_end'))
                else:
                    segments.append((start, sick[i-1] - 1, 'both_ends'))
                    start = -1
        if sick[-1] != n - 1:
            if start == -1:
                segments.append((sick[-1] + 1, n - 1, 'one_end'))
            else:
                segments.append((start, n - 1, 'both_ends'))

        # Calculate number of sequences for each segment
        total_sequences = 1
        m = len(segments)
        for segment in segments:
            start, end, connection = segment
            k = end - start + 1
            if connection == 'both_ends':
                sequences = pow_mod(2, k - 1)
            else:
                sequences = 1
            total_sequences = (total_sequences * sequences) % MOD

        # Multiply by the factorial of the number of segments
        total_sequences = (total_sequences * factorial(m)) % MOD

        return total_sequences
class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        def compute_flips(s, L):
            if L == 0:
                return float('inf')
            flips = 0
            if not s:
                return 0
            prev_char = s[0]
            current_run_length = 1
            for c in s[1:]:
                if c == prev_char:
                    current_run_length += 1
                    if current_run_length > L:
                        flips += 1
                        current_run_length = 1
                        prev_char = '1' if prev_char == '0' else '0'
                else:
                    current_run_length = 1
                    prev_char = c
            return flips

        # Compute the initial runs to find the maximum run length
        if not s:
            return 0
        runs = []
        current_char = s[0]
        count = 1
        for c in s[1:]:
            if c == current_char:
                count += 1
            else:
                runs.append(count)
                current_char = c
                count = 1
        runs.append(count)
        initial_max = max(runs) if runs else 0

        low, high = 1, initial_max
        res = initial_max

        while low <= high:
            mid = (low + high) // 2
            required = compute_flips(s, mid)
            if required <= numOps:
                res = mid
                high = mid - 1
            else:
                low = mid + 1

        return res
class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        def get_runs(s):
            runs = []
            n = len(s)
            if n == 0:
                return runs
            current_char = s[0]
            count = 1
            for i in range(1, n):
                if s[i] == current_char:
                    count += 1
                else:
                    runs.append(count)
                    current_char = s[i]
                    count = 1
            runs.append(count)
            return runs
        
        runs = get_runs(s)
        if not runs:
            return 0
        max_run = max(runs)
        low = 1
        high = max_run
        answer = max_run  # Initialize with the maximum possible
        
        while low <= high:
            mid = (low + high) // 2
            total_flips = 0
            for L in runs:
                if L <= mid:
                    continue
                else:
                    numerator = L - mid
                    denominator = mid + 1
                    required = (numerator + denominator - 1) // denominator
                    total_flips += required
                    if total_flips > numOps:
                        break  # No need to proceed further
            if total_flips <= numOps:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer
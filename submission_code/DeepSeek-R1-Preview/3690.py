class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        if not s:
            return 0
        
        # Precompute runs
        runs = []
        current = s[0]
        count = 1
        for c in s[1:]:
            if c == current:
                count += 1
            else:
                runs.append(count)
                current = c
                count = 1
        runs.append(count)
        
        if not runs:
            return 0
        
        max_run = max(runs)
        low = 1
        high = max_run
        answer = max_run
        
        while low <= high:
            mid = (low + high) // 2
            required = 0
            for run in runs:
                if run <= mid:
                    continue
                # Calculate the number of flips needed
                flips = (run + mid) // (mid + 1) - 1
                required += flips
                if required > numOps:
                    break
            if required <= numOps:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer
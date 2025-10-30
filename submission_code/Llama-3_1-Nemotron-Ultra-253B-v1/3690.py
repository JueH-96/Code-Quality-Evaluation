class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        # Split the string into runs of consecutive characters
        runs = []
        if not s:
            return 0
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
        
        # Determine the maximum run length in the original string
        max_run = max(runs) if runs else 0
        
        # Binary search for the minimal possible maximum length
        low, high = 1, max_run
        while low < high:
            mid = (low + high) // 2
            required = 0
            for r in runs:
                required += r // (mid + 1)
            if required <= numOps:
                high = mid
            else:
                low = mid + 1
        return low
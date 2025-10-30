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
        
        # Binary search for the minimum possible maximum length
        left, right = 1, max(runs)
        answer = right  # Initialize with the maximum possible value
        
        while left <= right:
            mid = (left + right) // 2
            required_flips = 0
            for run in runs:
                # Calculate the required flips for this run to have max segment <= mid
                required_flips += (run - 1) // (mid + 1)
            
            if required_flips <= numOps:
                # feasible, try to find a smaller X
                answer = mid
                right = mid - 1
            else:
                # not feasible, need to increase X
                left = mid + 1
        
        return answer
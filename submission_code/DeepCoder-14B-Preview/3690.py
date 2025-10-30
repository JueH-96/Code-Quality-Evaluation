class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        if not s:
            return 0
        
        # Split the string into runs of consecutive characters
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
        
        # Function to calculate the minimal number of operations required for a given maximum run length M
        def required_operations(M):
            if M == 0:
                return float('inf')
            total_ops = 0
            for length in runs:
                if length <= M:
                    continue
                # Calculate the minimal number of splits needed
                # Each split increases the number of parts by 1
                # The minimal number of parts needed is (length + M - 1) // M
                parts = (length + M - 1) // M
                total_ops += (parts - 1)
                if total_ops > numOps:
                    # Early exit if exceeding numOps
                    return float('inf')
            return total_ops
        
        # Binary search for the minimal possible maximum run length
        low = 1
        high = max(runs) if runs else 0
        answer = high
        
        while low <= high:
            mid = (low + high) // 2
            req = required_operations(mid)
            if req <= numOps:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer
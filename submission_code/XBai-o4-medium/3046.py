class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = float('inf')
        
        # Check all four possible endings for divisibility by 25
        for ending in ["00", "25", "50", "75"]:
            a, b = ending[0], ending[1]
            # Find the last occurrence of b
            j = None
            for i in range(n-1, -1, -1):
                if num[i] == b:
                    j = i
                    break
            if j is None:
                continue  # No occurrence of b found
            # Find the last occurrence of a before j
            i_found = None
            for k in range(j-1, -1, -1):
                if num[k] == a:
                    i_found = k
                    break
            if i_found is not None:
                steps = n - i_found - 2
                if steps < min_ops:
                    min_ops = steps
        
        # Check the case of having a single zero
        if '0' in num:
            single_zero_steps = n - 1
            if single_zero_steps < min_ops:
                min_ops = single_zero_steps
        
        # Check the case of deleting all digits
        delete_all_steps = n
        if delete_all_steps < min_ops:
            min_ops = delete_all_steps
        
        return min_ops if min_ops != float('inf') else 0
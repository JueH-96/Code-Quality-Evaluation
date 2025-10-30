import bisect
from collections import defaultdict

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Compute prefix sum array of zeros
        prefix = [0]
        current_zeros = 0
        for c in s:
            if c == '0':
                current_zeros += 1
            prefix.append(current_zeros)
        
        # Build a dictionary to store indices of each prefix value
        value_indices = defaultdict(list)
        for idx, val in enumerate(prefix):
            value_indices[val].append(idx)
        
        # Count all substrings with no zeros (all ones)
        count = 0
        current_length = 0
        for c in s:
            if c == '1':
                current_length += 1
            else:
                if current_length > 0:
                    count += current_length * (current_length + 1) // 2
                    current_length = 0
        if current_length > 0:
            count += current_length * (current_length + 1) // 2
        
        total = count
        
        # Check for z from 1 to maximum feasible z (up to 200)
        max_z = 200
        for z in range(1, max_z + 1):
            L = z * (z + 1)
            if L > n:
                continue  # No possible substrings of length >= L
            
            for j in range(len(prefix)):
                target = prefix[j] - z
                max_i = j - L
                if max_i < 0:
                    continue
                # Find number of indices in value_indices[target] that are <= max_i
                indices = value_indices.get(target, [])
                cnt = bisect.bisect_right(indices, max_i)
                total += cnt
        
        return total
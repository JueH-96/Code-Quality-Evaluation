class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        
        # Find all positions of zeros
        zeros_pos = [i for i, char in enumerate(s) if char == '0']
        m = len(zeros_pos)
        
        # Function to compute z_max
        z_max = (int((1 + 4 * n)**0.5) - 1) // 2
        
        # Count substrings with z zeros and l <= z^2 + z -1
        invalid_count = 0
        
        # For z = 0
        # Find sequences of consecutive ones
        left = 0
        while left < n:
            if s[left] == '1':
                right = left
                while right < n and s[right] == '1':
                    right += 1
                m = right - left
                invalid_count += m * (m + 1) // 2
                left = right
            else:
                left += 1
        
        # For z >= 1
        for z in range(1, z_max + 1):
            for k in range(m - z + 1):
                p0 = zeros_pos[k]
                pz = zeros_pos[k + z - 1]
                l_min = z**2 + z - 1
                if pz - p0 + 1 <= l_min:
                    # Number of starting positions
                    start = max(0, pz - l_min)
                    # Number of ending positions
                    end = min(n - 1, p0 + l_min - 1)
                    # Total substrings
                    total_substrings = (p0 - start + 1) * (end - pz + 1)
                    invalid_count += total_substrings
        
        # Substrings with o >= z^2
        result = total - invalid_count
        return result
class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import defaultdict

        # Step 1: Record frequency and positions of each character
        freq = defaultdict(int)
        positions = defaultdict(list)
        for idx, char in enumerate(s):
            freq[char] += 1
            positions[char].append(idx)
        
        # Step 2: Find the maximum frequency
        k = max(freq.values())
        
        # Step 3: For each character with frequency >=k, take the last (f - (k-1)) positions
        remaining_positions = []
        for char, pos_list in positions.items():
            f = freq[char]
            if f >= k:
                # Take the last (f - (k-1)) positions
                num_remaining = f - (k - 1)
                remaining_positions.extend(pos_list[-num_remaining:])
        
        # Step 4: Sort the remaining positions and build the resulting string
        remaining_positions.sort()
        result = ''.join([s[pos] for pos in remaining_positions])
        
        return result
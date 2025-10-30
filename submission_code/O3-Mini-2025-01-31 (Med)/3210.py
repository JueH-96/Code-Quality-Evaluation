class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        vowels_set = set("aeiou")
        
        # Build prefix sums, where prefix[i] is the "balance" up to but not including s[i]
        # Use +1 for vowel, -1 for consonant.
        prefix = [0] * (n + 1)
        for i in range(n):
            if s[i] in vowels_set:
                prefix[i + 1] = prefix[i] + 1
            else:
                prefix[i + 1] = prefix[i] - 1
        
        # Group indices by prefix sum value.
        groups = {}
        for i, val in enumerate(prefix):
            if val not in groups:
                groups[val] = []
            groups[val].append(i)
        
        # Precompute a boolean table for valid n values (n = number of vowels or consonants in the substring)
        # We need to check condition (n*n) % k == 0, for n >= 1 up to maximum possible substring half length.
        # Maximum possible half-length is n//2.
        max_half = n // 2
        valid_n = [False] * (max_half + 1)
        for num in range(1, max_half + 1):
            if (num * num) % k == 0:
                valid_n[num] = True
        
        count = 0
        # For each group, we consider all pairs (i, j) with i < j, where prefix[j] == prefix[i].
        # For the substring from i to j-1:
        #   length = j - i, and since the substring is balanced, we must have (j - i) is even.
        #   Let n = (j - i) // 2. Check if valid_n[n] is True.
        for indices in groups.values():
            # Only consider pairs (i, j) where difference is even; note that j-i is even exactly when i and j have the same parity.
            even_positions = []
            odd_positions = []
            for idx in indices:
                if idx % 2 == 0:
                    even_positions.append(idx)
                else:
                    odd_positions.append(idx)
                    
            # process each parity group
            for group in (even_positions, odd_positions):
                sz = len(group)
                # Brute force all pairs in this group.
                # Since the total length is at most 1001, and each group will be relatively small, this is acceptable.
                for i in range(sz):
                    for j in range(i+1, sz):
                        diff = group[j] - group[i]  # this is the substring length
                        # if diff is 0 then substring is empty, so you need diff >= 2 (non-empty balanced substring has even length and at least 2 characters)
                        if diff > 0:
                            half = diff // 2  # number of vowels equals half the length.
                            # Check our condition: (half^2) % k == 0.
                            if valid_n[half]:
                                count += 1
        return count


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    s1 = "baeyh"
    k1 = 2
    print(sol.beautifulSubstrings(s1, k1))  # Expected output: 2
    
    # Example 2
    s2 = "abba"
    k2 = 1
    print(sol.beautifulSubstrings(s2, k2))  # Expected output: 3
    
    # Example 3
    s3 = "bcdf"
    k3 = 1
    print(sol.beautifulSubstrings(s3, k3))  # Expected output: 0
class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute prefix sums for each character
        prefix = [[0] * 26]
        for i in range(n):
            new_entry = prefix[-1].copy()
            char_idx = ord(s[i]) - ord('a')
            new_entry[char_idx] += 1
            prefix.append(new_entry)
        
        # Function to get counts between start and end (exclusive)
        def get_counts(start, end):
            return [prefix[end][i] - prefix[start][i] for i in range(26)]
        
        # Find all divisors of n
        divisors = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        divisors = sorted(divisors)
        
        # Check each divisor in ascending order
        for k in divisors:
            m = n // k
            first_counts = get_counts(0, k)
            valid = True
            for i in range(1, m):
                start = i * k
                end = start + k
                current_counts = get_counts(start, end)
                if current_counts != first_counts:
                    valid = False
                    break
            if valid:
                return k
        
        return n  # This line is theoretically unreachable
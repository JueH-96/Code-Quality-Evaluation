class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute the prefix array
        prefix = []
        current = [0] * 26
        prefix.append(list(current))
        for c in s:
            idx = ord(c) - ord('a')
            new_current = current.copy()
            new_current[idx] += 1
            prefix.append(new_current)
            current = new_current
        
        # Get all divisors of n in sorted order
        divisors = self.get_divisors(n)
        
        for L in divisors:
            if n % L != 0:
                continue  # This should not happen as divisors are correct
            
            # Check if all chunks of length L have the same counts as the first chunk
            first_counts = prefix[L]
            valid = True
            for start in range(L, n, L):
                end = start + L
                # Compute current counts by subtracting prefix[start] from prefix[end]
                current_counts = [prefix[end][i] - prefix[start][i] for i in range(26)]
                if current_counts != first_counts:
                    valid = False
                    break
            if valid:
                return L
        return n  # This line is a fallback, but the last divisor is n itself
    
    def get_divisors(self, n):
        divisors = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        return sorted(divisors)
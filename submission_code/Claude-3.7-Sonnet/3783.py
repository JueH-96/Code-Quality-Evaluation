class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # Calculate factorial values we'll need
        fact = [1]
        for i in range(1, n+1):
            fact.append(fact[-1] * i)
            
        # Calculate total alternating permutations
        m = n // 2
        if n % 2 == 0:
            total = 2 * fact[m] ** 2
        else:
            total = fact[m+1] * fact[m]
            
        if k > total:
            return []
            
        # Separate odd and even numbers
        odd = [i for i in range(1, n+1) if i % 2 == 1]
        even = [i for i in range(1, n+1) if i % 2 == 0]
        
        # Determine pattern (odd-first or even-first)
        pattern1_count = fact[m] ** 2 if n % 2 == 0 else total
        
        if k <= pattern1_count:
            # Pattern 1: odd in odd positions, even in even positions
            odd_perm = self.kth_permutation(odd, (k-1)//fact[m]+1, fact)
            even_perm = self.kth_permutation(even, (k-1)%fact[m]+1, fact)
            return self.interleave(odd_perm, even_perm)
        else:
            # Pattern 2: even in odd positions, odd in even positions
            k -= pattern1_count
            even_perm = self.kth_permutation(even, (k-1)//fact[m]+1, fact)
            odd_perm = self.kth_permutation(odd, (k-1)%fact[m]+1, fact)
            return self.interleave(even_perm, odd_perm)
    
    def kth_permutation(self, nums, k, fact):
        """Get the kth permutation of nums in lexicographic order."""
        n = len(nums)
        result = []
        nums = nums.copy()
        k -= 1  # Convert to 0-indexed
        
        for i in range(n):
            idx = k // fact[n-1-i]
            result.append(nums.pop(idx))
            k %= fact[n-1-i]
            
        return result
    
    def interleave(self, list1, list2):
        """Interleave two lists."""
        result = []
        for i in range(max(len(list1), len(list2))):
            if i < len(list1):
                result.append(list1[i])
            if i < len(list2):
                result.append(list2[i])
        return result
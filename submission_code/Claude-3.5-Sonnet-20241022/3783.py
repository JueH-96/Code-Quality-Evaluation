class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        def is_valid(perm):
            # Check if permutation is alternating (no two adjacent numbers are both odd or both even)
            for i in range(len(perm)-1):
                if perm[i] % 2 == perm[i+1] % 2:
                    return False
            return True
        
        def generate_next_permutation(perm):
            # Find the largest index k such that perm[k] < perm[k + 1]
            k = len(perm) - 2
            while k >= 0 and perm[k] >= perm[k + 1]:
                k -= 1
                
            # If no such index exists, return None (no more permutations)
            if k < 0:
                return None
            
            # Find the largest index l such that perm[k] < perm[l]
            l = len(perm) - 1
            while l >= 0 and perm[k] >= perm[l]:
                l -= 1
                
            # Swap the k-th and l-th elements
            perm[k], perm[l] = perm[l], perm[k]
            
            # Reverse the sequence from k+1 up to and including the final element
            left = k + 1
            right = len(perm) - 1
            while left < right:
                perm[left], perm[right] = perm[right], perm[left]
                left += 1
                right -= 1
                
            return perm
        
        # Initialize first permutation
        current = list(range(1, n + 1))
        count = 0
        
        # Generate permutations until we find the k-th alternating one
        while True:
            if is_valid(current):
                count += 1
                if count == k:
                    return current
            
            current = generate_next_permutation(current[:])
            if current is None:
                return []  # k is out of range
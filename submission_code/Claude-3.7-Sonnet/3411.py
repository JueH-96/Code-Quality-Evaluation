class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        # Function to count set bits in an integer
        def popcount(x):
            return bin(x).count('1')
            
        # Function to calculate sum of bits set in all integers from 1 to n
        def prefix_popcount(n):
            if n <= 0:
                return 0
                
            # Find largest power of 2 less than or equal to n
            k = n.bit_length() - 1
            
            # Count bits for numbers 1 to 2^k-1
            result = k * (1 << (k - 1)) if k > 0 else 0
            
            # Count bits for remaining numbers (2^k to n)
            remaining = n - (1 << k) + 1
            result += remaining  # High bit is always 1
            result += prefix_popcount(remaining - 1)  # Count remaining bits
            
            return result
        
        # Function to find the element at a specific index in big_nums
        def get_element(idx):
            # Binary search to find which integer's powerful array contains this index
            left, right = 1, 10**16
            while left < right:
                mid = (left + right) // 2
                count_before = prefix_popcount(mid - 1)
                count_at = prefix_popcount(mid)
                
                if count_before <= idx < count_at:
                    # Found the integer
                    k = mid
                    pos = idx - count_before
                    
                    # Find which bit position corresponds to this index
                    count = 0
                    for bit_pos in range(k.bit_length()):
                        if (k >> bit_pos) & 1:
                            if count == pos:
                                return 1 << bit_pos
                            count += 1
                            
                    return -1  # Should not happen
                    
                elif count_before > idx:
                    right = mid
                else:
                    left = mid + 1
            
            return -1  # Should not happen
        
        # Process each query
        results = []
        for from_i, to_i, mod_i in queries:
            product = 1
            for i in range(from_i, to_i + 1):
                element = get_element(i)
                product = (product * element) % mod_i
            results.append(product)
            
        return results
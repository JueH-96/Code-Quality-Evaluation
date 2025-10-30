class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def get_powerful_array(n):
            if n == 0:
                return []
            powers = []
            power = 1
            while power <= n:
                if n & power:
                    powers.append(power)
                power <<= 1
            return powers
            
        def get_element_at_index(idx):
            # Binary search to find which number this index belongs to
            left, right = 1, 10**15
            while left < right:
                mid = (left + right) // 2
                total_len = 0
                for i in range(1, mid + 1):
                    total_len += len(get_powerful_array(i))
                if total_len <= idx:
                    left = mid + 1
                else:
                    right = mid
                    
            num = left
            # Find position within the powerful array
            prev_len = 0
            for i in range(1, num):
                prev_len += len(get_powerful_array(i))
            pos = idx - prev_len
            return get_powerful_array(num)[pos]
        
        result = []
        for start, end, mod in queries:
            product = 1
            for i in range(start, end + 1):
                product = (product * get_element_at_index(i)) % mod
            result.append(product)
            
        return result
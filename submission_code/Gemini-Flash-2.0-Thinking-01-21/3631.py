class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        n = int(s, 2)
        if n <= 1:
            return 0
        
        def count_set_bits(x):
            count = 0
            while x > 0:
                if x % 2 == 1:
                    count += 1
                x //= 2
            return count
            
        def is_k_reducible(x, k_val):
            current_x = x
            for _ in range(k_val):
                current_x = count_set_bits(current_x)
                if current_x == 1:
                    return True
            return False
            
        count = 0
        for x in range(1, n):
            if is_k_reducible(x, k):
                count += 1
        return count
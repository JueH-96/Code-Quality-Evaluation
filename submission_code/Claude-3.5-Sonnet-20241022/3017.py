class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(num, k):
            # Convert number to string to count digits
            num_str = str(num)
            even_count = sum(1 for d in num_str if int(d) % 2 == 0)
            odd_count = len(num_str) - even_count
            
            # Check both conditions
            return even_count == odd_count and num % k == 0
        
        # Count beautiful numbers in range
        count = 0
        # Start from first multiple of k >= low
        start = ((low + k - 1) // k) * k
        
        for num in range(start, high + 1, k):
            if is_beautiful(num, k):
                count += 1
                
        return count
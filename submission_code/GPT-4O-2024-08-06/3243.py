class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # Convert s to an integer for comparison
        s_int = int(s)
        # Length of the suffix s
        len_s = len(s)
        
        # Initialize the count of powerful integers
        count = 0
        
        # Iterate over the possible prefixes
        # We need to consider numbers that can be formed by prefixing s with digits <= limit
        # The number of digits in the prefix can be from 0 to the number of digits in finish - len_s
        for prefix_length in range(0, len(str(finish)) - len_s + 1):
            # Generate all possible prefixes of the given length
            # We use a queue to generate numbers with digits <= limit
            queue = ['']
            while queue:
                prefix = queue.pop(0)
                
                # If the current prefix length is equal to the desired prefix length, check the number
                if len(prefix) == prefix_length:
                    # Form the number by appending the suffix s
                    num_str = prefix + s
                    num = int(num_str)
                    
                    # Check if the number is within the range and has valid digits
                    if start <= num <= finish:
                        # Check if all digits in the number are <= limit
                        if all(int(d) <= limit for d in num_str):
                            count += 1
                else:
                    # If the prefix is not yet of the desired length, extend it with digits <= limit
                    for digit in range(0, limit + 1):
                        queue.append(prefix + str(digit))
        
        return count
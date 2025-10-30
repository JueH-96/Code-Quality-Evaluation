class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        low, high = int(low), int(high)
        
        from collections import deque
        
        # Initialize the queue with numbers from 1 to 9 (0 is not included as it can lead to leading zeros)
        queue = deque(range(1, 10))
        count = 0
        
        # Special case for 0, check if it's in the range
        if low == 0:
            count += 1
        
        while queue:
            current = queue.popleft()
            
            # If current is within the range, increment the count
            if low <= current <= high:
                count = (count + 1) % MOD
            
            # If current is greater than high, no need to process further
            if current > high:
                continue
            
            # Get the last digit of the current number
            last_digit = current % 10
            
            # Generate next possible stepping numbers
            if last_digit == 0:
                next_number = current * 10 + 1
                if next_number <= high:
                    queue.append(next_number)
            elif last_digit == 9:
                next_number = current * 10 + 8
                if next_number <= high:
                    queue.append(next_number)
            else:
                next_number1 = current * 10 + (last_digit - 1)
                next_number2 = current * 10 + (last_digit + 1)
                if next_number1 <= high:
                    queue.append(next_number1)
                if next_number2 <= high:
                    queue.append(next_number2)
        
        return count
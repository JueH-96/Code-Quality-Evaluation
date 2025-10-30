class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        from collections import deque

        MOD = 10**9 + 7

        # Comparison function for number strings
        def compare(num1, num2):
            if len(num1) > len(num2):
                return 1
            elif len(num1) < len(num2):
                return -1
            else:
                if num1 > num2:
                    return 1
                elif num1 < num2:
                    return -1
                else:
                    return 0

        # Initialize queue with single-digit stepping numbers
        queue = deque(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        count = 0

        while queue:
            num = queue.popleft()

            # Check if the current number is greater than high
            if compare(num, high) == 1:
                continue

            # Check if the current number is less than low
            if compare(num, low) == -1:
                continue

            # If within range, count it
            count = (count + 1) % MOD

            # Generate next stepping numbers if length is less than 100
            if len(num) < 100:
                last_digit = int(num[-1])
                next_digits = []
                if last_digit > 0:
                    next_digits.append(str(last_digit - 1))
                if last_digit < 9:
                    next_digits.append(str(last_digit + 1))
                for nd in next_digits:
                    queue.append(num + nd)

        return count
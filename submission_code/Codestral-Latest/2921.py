class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def is_stepping_number(num: str) -> bool:
            if len(num) == 1:
                return True
            for i in range(1, len(num)):
                if abs(int(num[i]) - int(num[i - 1])) != 1:
                    return False
            return True

        def count_stepping_numbers_in_range(low: str, high: str) -> int:
            count = 0
            low_int = int(low)
            high_int = int(high)

            for num in range(low_int, high_int + 1):
                if is_stepping_number(str(num)):
                    count += 1
                    count %= MOD

            return count

        return count_stepping_numbers_in_range(low, high)
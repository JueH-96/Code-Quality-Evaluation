class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def is_stepping(num):
            for i in range(len(num) - 1):
                if abs(int(num[i]) - int(num[i + 1])) != 1:
                    return False
            return True

        low_int = int(low)
        high_int = int(high)

        count = 0
        for num in range(low_int, high_int + 1):
            if is_stepping(str(num)):
                count += 1

        return count % MOD
class Solution:
    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        dp = {}

        def solve(index, prev_drink):
            if index == n:
                return 0
            
            if (index, prev_drink) in dp:
                return dp[(index, prev_drink)]

            ans = 0
            
            # Option 1: Continue with drink A
            if prev_drink != 'A':
                ans = max(ans, energyDrinkA[index] + solve(index + 1, 'A'))
            else:
                ans = max(ans, solve(index + 1, None))

            # Option 2: Continue with drink B
            if prev_drink != 'B':
                ans = max(ans, energyDrinkB[index] + solve(index + 1, 'B'))
            else:
                ans = max(ans, solve(index + 1, None))
            
            dp[(index, prev_drink)] = ans
            return ans

        return solve(0, None)
from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [0] * n
        # The idea is that to reach any position i, we need to pay at least the minimum cost among
        # all persons with indices 0 to i. Because once we pay that smallest cost, we can swap with
        # any person behind that paid person for free.
        current_min = float('inf')
        for i in range(n):
            current_min = min(current_min, cost[i])
            answer[i] = current_min
        return answer

# Example to test the solution:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minCosts([5, 3, 4, 1, 3, 2]))  # Expected output: [5, 3, 3, 1, 1, 1]
    print(sol.minCosts([1, 2, 4, 6, 7]))        # Expected output: [1, 1, 1, 1, 1]
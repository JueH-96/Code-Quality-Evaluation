from typing import List
import itertools

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        best = float('inf')
        # Try all orders of breaking locks
        for perm in itertools.permutations(strength):
            time_spent = 0
            X = 1
            # early prune
            for s in perm:
                # minutes needed to reach at least s energy at rate X per minute
                t = (s + X - 1) // X
                time_spent += t
                if time_spent >= best:
                    break
                X += K
            else:
                # completed all locks
                if time_spent < best:
                    best = time_spent
        return best

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.findMinimumTime([3, 4, 1], 1))  # Output: 4
    print(sol.findMinimumTime([2, 5, 4], 2))  # Output: 5
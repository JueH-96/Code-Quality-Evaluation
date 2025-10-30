from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        valid_count = 0
        
        # Try every starting index that has a 0
        for start in range(n):
            if nums[start] != 0:
                continue
            # Try both possible initial directions: 1 (right) and -1 (left)
            for initial_direction in [1, -1]:
                # Use a copy of nums for this simulation
                arr = nums[:]  
                curr = start
                direction = initial_direction
                # Continue simulation until the index goes out of the array bounds
                while 0 <= curr < n:
                    if arr[curr] == 0:
                        curr += direction
                    else:
                        # Decrement the positive number, reverse direction, and take a step in that direction
                        arr[curr] -= 1
                        direction = -direction
                        curr += direction

                # If at the end all elements are 0, this starting selection is valid.
                if all(x == 0 for x in arr):
                    valid_count += 1
                    
        return valid_count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countValidSelections([1, 0, 2, 0, 3]))  # Expected output: 2
    print(sol.countValidSelections([2, 3, 4, 0, 4, 1, 0]))  # Expected output: 0
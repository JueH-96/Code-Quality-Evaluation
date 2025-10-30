class Solution:
    def largestPerimeter(self, nums: list) -> int:
        # Sort the numbers in descending order so that the first element is the largest.
        a = sorted(nums, reverse=True)
        total = sum(a)
        n = len(a)
        # Use a pointer to simulate the removal of the largest element if necessary.
        # Instead of physically removing elements (which can be inefficient),
        # we adjust the total sum and the starting index.
        i = 0
        while n - i >= 3:  # At least 3 sides are needed for a polygon.
            # For any polygon, the condition is that the sum of all sides except the longest
            # must be strictly greater than the longest side.
            # Here, a[i] is the current largest side in our current subset.
            if total > 2 * a[i]:
                return total  # We have found the largest valid polygon perimeter.
            # Otherwise, remove the current largest side by updating the total and incrementing i.
            total -= a[i]
            i += 1
        # If we end up with less than 3 sides, we cannot form a polygon.
        return -1

# Example test cases:
if __name__ == '__main__':
    sol = Solution()
    print(sol.largestPerimeter([5,5,5]))           # Expected output: 15
    print(sol.largestPerimeter([1,12,1,2,5,50,3]))  # Expected output: 12
    print(sol.largestPerimeter([5,5,50]))           # Expected output: -1
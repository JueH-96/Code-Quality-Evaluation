class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # Initialize an array with the first number 1
        arr = [1]
        # Initialize the sum to 1
        total = 1
        # Iterate from 2 to n
        for i in range(2, n+1):
            # If i is not in the array and i and the last number in the array do not sum to k, add i to the array and update the sum
            if i not in arr and total + i != k:
                arr.append(i)
                total += i
        # Return the sum
        return total
from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        # Explanation:
        # The imbalance number for an array is defined as the number of “gaps” in the sorted array of distinct values.
        # More precisely, if the distinct sorted values form groups of consecutive numbers,
        # then the imbalance equals: (number of groups – 1).
        # Equivalently, if the distinct set of numbers in a subarray is D and
        # if we define “consecutive pairs” among D as any pair (x, x+1) that appears,
        # then the number of groups is: |D| – (number of consecutive pairs)
        # and imbalance = (|D| – (number of consecutive pairs)) – 1.
        # Rearranging, imbalance = |D| – 1 – (number of consecutive pairs).
        #
        # To solve the problem we want the sum over all subarrays:
        #   sum_{subarray} (distinct_count - 1 - (# of consecutive pairs))
        # where “distinct_count” means the size of the distinct set in that subarray.
        #
        # A direct way to do this is to iterate over all subarrays.
        # Since n <= 1000, the total subarrays count is about (n*(n+1))//2,
        # which in worst case is about 500K iterations. In each subarray we update a frequency array.
        # Because all nums[i] are in the range [1, n], we can use a list of size n+2.
        #
        # The update when adding a new element x is as follows:
        #   • If x is seen for the first time (frequency==0) then:
        #       - Increase distinct count.
        #       - Check the two potential consecutive neighbors:
        #           If x-1 is already present (frequency > 0), then the pair (x-1, x)
        #           now appears and we increase our consecutive-pairs count.
        #           Similarly, if x+1 is already present, then the pair (x, x+1) appears.
        #   • (If x was already in the subarray, duplicates do not affect distinct count
        #       nor the count of consecutive neighbors, since we count each pair once.)
        #
        # Then for each subarray [i, j] we add:
        #    imbalance = (distinct - 1) - (consecutive pairs)
        # (For a one-element subarray the imbalance is 0.)
        #
        # Finally, we return the total sum.
        n = len(nums)
        total = 0
        # Iterate each starting index of a subarray.
        for i in range(n):
            # frequency array for elements from 1 to n (we allocate n+2 so that x+1 is in range)
            freq = [0] * (n + 2)  
            distinct = 0
            consecutive = 0  # count of consecutive pairs found in the distinct set
            # Extend the subarray from i to j
            for j in range(i, n):
                x = nums[j]
                if freq[x] == 0:
                    distinct += 1
                    # When x appears for the first time, its neighbor x-1 might have been seen.
                    if x - 1 >= 1 and freq[x - 1] > 0:
                        consecutive += 1
                    # Similarly, if neighbor x+1 is already seen, then the pair (x, x+1) is formed.
                    if x + 1 <= n and freq[x + 1] > 0:
                        consecutive += 1
                freq[x] += 1
                # For the current subarray [i, j], the imbalance is: 
                #   (distinct - 1) - (number of consecutive pairs)
                total += (distinct - 1 - consecutive)
        return total

# The following code is for local testing and uses the provided "starter code" format.
if __name__ == '__main__':
    # To allow both running in a script environment and using standard input, we include a simple runner.
    import sys
    data = sys.stdin.read().strip().split()
    if data:
        # Assume all the data represents the list of numbers.
        # For example: "2 3 1 4" for the first example.
        nums = list(map(int, data))
        sol = Solution()
        sys.stdout.write(str(sol.sumImbalanceNumbers(nums)))
    else:
        # Local tests:
        sol = Solution()
        print(sol.sumImbalanceNumbers([2, 3, 1, 4]))   # Expected output: 3
        print(sol.sumImbalanceNumbers([1, 3, 3, 3, 5])) # Expected output: 8
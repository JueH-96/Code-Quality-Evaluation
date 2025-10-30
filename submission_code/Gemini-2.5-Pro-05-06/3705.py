import collections
from typing import List

class Solution:
  def largestInteger(self, nums: List[int], k: int) -> int:
    n = len(nums)
    
    # This map will store the count of k-sized subarrays each number appears in.
    # Key: number (from nums), Value: count of appearances.
    num_subarray_counts = collections.defaultdict(int)

    # Iterate through all possible starting positions of a subarray of size k.
    # A subarray nums[i:i+k] starts at index i and ends at index i+k-1.
    # The last possible starting index is n-k. So i ranges from 0 to n-k (inclusive).
    # The loop range(n - k + 1) goes from 0 up to (n-k+1)-1 = n-k.
    for i in range(n - k + 1):
        # Extract the current subarray. Slicing takes O(k) time.
        current_subarray_slice = nums[i : i+k] 
        
        # Find all unique numbers in this current_subarray_slice.
        # We only care if a number *appears* in the subarray, not its frequency within it.
        # Converting to a set handles duplicates within the subarray automatically.
        # This operation takes O(k) time on average.
        distinct_elements_in_current_subarray = set(current_subarray_slice)
        
        # For each unique number found in this subarray, increment its appearance count
        # in num_subarray_counts. The number of distinct elements is at most k.
        # Dictionary update is O(1) on average.
        for num_val in distinct_elements_in_current_subarray:
            num_subarray_counts[num_val] += 1
    
    # After processing all subarrays, num_subarray_counts contains all numbers that 
    # appeared in at least one k-sized subarray, along with how many such subarrays 
    # they appeared in.
    
    # Now, find the "almost missing" integers.
    # An integer x is "almost missing" if it appears in exactly one subarray of size k.
    # We are looking for the largest such integer.
    
    # Initialize with -1, as per problem specification if no such integer exists.
    # Since all numbers in nums are non-negative (0 to 50 as per constraints), 
    # any "almost missing" integer found will be >= 0. 
    # So, if largest_almost_missing remains -1, it means no such integer was found.
    largest_almost_missing = -1 

    # Iterate through all numbers (and their counts) that appeared in any subarray.
    # The number of entries in num_subarray_counts is at most D (number of distinct values in nums),
    # which is at most n, or 51 (given the range of values 0-50).
    for num_val, count in num_subarray_counts.items():
        if count == 1:
            # This number 'num_val' is "almost missing".
            # We need the largest such number. Update if 'num_val' is larger.
            if num_val > largest_almost_missing:
                largest_almost_missing = num_val
            # An equivalent way using max():
            # largest_almost_missing = max(largest_almost_missing, num_val)
            # This works because num_val >= 0.
    
    return largest_almost_missing
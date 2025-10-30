from typing import List
import heapq

MOD = 10**9 + 7

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        # Explanation:
        # The allowed operation never changes the overall count of ones in any bit position.
        # In each operation when you choose indices i and j,
        # the pair (nums[i], nums[j]) is transformed into (nums[i]&nums[j], nums[i]|nums[j]).
        # For any bit b the sum of that bit (0 or 1) over the two numbers remains the same.
        #
        # Therefore, for each bit b the total number of ones in the array remains constant.
        # In the final configuration (after arbitrary operations) we are allowed to re-distribute these
        # ones among entries subject to the restriction that any chosen number (an integer) can either have
        # bit b set (1) or not (0) – in other words, each index gets at most one “copy” of bit b.
        #
        # We are allowed to pick k elements from the final array. Since we want to maximize the sum of squares,
        # and the square function is convex, it is best to “concentrate” the available ones among these k numbers
        # as much as possible.
        #
        # For any bit b:
        #   • If its total count (call it cnt) is at least k then no matter what we do,
        #     every one of the k chosen numbers must have that bit set. (If not, we could “push” it from
        #     an unchosen index into one of the k chosen indices using the allowed operations.)
        #   • If cnt < k then we have exactly cnt copies of the bit
        #     (each copy must go to a distinct chosen index, since a number cannot have the same bit twice).
        #
        # Let the value of bit b be v = 2^b.
        # For bits with cnt >= k, every chosen index gets that bit.
        # Thus, each chosen number gets an obligatory “baseline” contribution.
        # Let baseline B = sum_{b: cnt >= k} 2^b.
        #
        # For bits with cnt < k (and cnt > 0) we have cnt copies that we can assign arbitrarily
        # among the k bins (each bin can get the bit at most once).
        # In order to maximize the total sum of squares,
        # we want to “concentrate” these extra contributions in as few bins as possible.
        # In other words, given extra items (each valued 2^b) that must go to distinct bins for a given bit b,
        # we want to assign each such copy to the bin with the highest current value.
        # (The marginal gain of adding an extra v to a number x is ( (x+v)^2 - x^2 ) = 2*x*v + v^2,
        # which is larger when x is larger.)
        #
        # We thus simulate the following:
        #   1. Compute for each bit (0-indexed) the count of ones.
        #   2. Every bit b with count >= k forces a contribution of 2^b to every one of the k chosen indices.
        #      That is our baseline B.
        #   3. For each bit b with 0 < count < k, we have count copies (value 2^b each)
        #      that we are free to assign to k chosen indices subject to at most one per bin.
        #      We do this greedily in order of decreasing 2^b.
        #      For a bit type with count copies, we take the top "count" bins (the bins with the highest
        #      current total) and add 2^b to them.
        #
        # Once we have “distributed” every available bit, the sum of squares of the k chosen numbers is maximized.
      
        n = len(nums)
        # Determine maximum bit needed. (nums[i] <= 10^9 so up to 30 or 31 bits)
        max_bit = max(nums).bit_length() if nums else 0
        
        # Count ones for each bit position.
        counts = [0] * (max_bit + 1)
        for num in nums:
            for b in range(max_bit + 1):
                if num & (1 << b):
                    counts[b] += 1
        
        baseline = 0
        # For bits with total count < k (but > 0) we have extra copies we can assign.
        extras = []  # Will hold tuples of (value, available copies)
        for b in range(max_bit + 1):
            v = 1 << b
            if counts[b] >= k:
                # Every chosen number must have this bit.
                baseline += v
            elif counts[b] > 0:
                # We have counts[b] copies (each must go to a distinct bin)
                extras.append((v, counts[b]))
                
        # To maximize the square sum we want to add higher value bits first.
        extras.sort(key=lambda x: x[0], reverse=True)
        
        # Initialize the k chosen bins with the baseline value.
        bins = [baseline] * k
        
        # We use a max-heap to always add an extra bit to the bin with the highest current value.
        # Python’s heapq implements a min-heap, so we store negatives.
        heap = [-x for x in bins]
        heapq.heapify(heap)
        
        # Process each extra bit type.
        for v, count_available in extras:
            # For a given bit, we have count_available copies.
            # They must go to count_available distinct bins.
            # Get the top count_available bins, update them, and push them back.
            temp = []
            for _ in range(count_available):
                # Note: count_available is always <= k (since counts[b] < k)
                cur = -heapq.heappop(heap)
                temp.append(cur + v)
            for new_val in temp:
                heapq.heappush(heap, -new_val)
        
        # Now the heap contains the k chosen numbers.
        # Compute the answer = sum(x^2) mod MOD.
        ans = 0
        while heap:
            x = -heapq.heappop(heap)
            ans = (ans + (x % MOD) * (x % MOD)) % MOD
        return ans

# For local testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    nums = [2, 6, 5, 8]
    k = 2
    print(sol.maxSum(nums, k))  # Expected output: 261

    # Example 2:
    nums = [4, 5, 4, 7]
    k = 3
    print(sol.maxSum(nums, k))  # Expected output: 90
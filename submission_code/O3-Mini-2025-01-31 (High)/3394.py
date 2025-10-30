class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # When the array has just one element, the only possibility is [x].
        if n == 1:
            return x

        # Every element in the array must include all 1‐bits of x.
        # In other words, every number is of the form: a = x + k,
        # where k’s binary representation has bits only in the positions 
        # where x has 0’s (we say these positions are "free" to choose).
        # 
        # Also, for the bitwise AND of all a's to be exactly x, it suffices that
        # at least one element “clears” each free bit (i.e. has a 0 there). Since
        # x itself is used (and it is x + 0), it clears all free bits.
        #
        # To minimize the maximum element (nums[n-1]), it is optimal to choose the
        # n smallest numbers from the set: { x + k : k is a non‐negative integer and 
        # k uses only free bits }.
        #
        # Notice that if you restrict k to use only a fixed set of r free bit positions,
        # then k can take any value from 0 to 2^r - 1 (the mapping is one‐to‐one).
        # So, if we choose the r smallest free bit positions such that 2^r >= n,
        # then the n smallest valid k values are exactly the numbers 0, 1, 2, …, n-1,
        # but “translated” through those bit positions.
        #
        # Our task is to find the smallest such r (which is r = ceil(log2(n))) and
        # the corresponding r free bit positions. Then, writing (n - 1) in binary
        # (using r bits) and mapping the j–th bit (0-indexed) to the free bit position
        # found at free_bits[j] gives us the desired increment k. Finally, the answer
        # is x + k.
        
        # Determine the minimum number of free bit positions needed.
        r = (n - 1).bit_length()  # smallest r with 2^r >= n

        # Find the r smallest free bit positions (positions where x has 0).
        free_bits = []
        i = 0
        while len(free_bits) < r:
            if ((x >> i) & 1) == 0:
                free_bits.append(i)
            i += 1
        
        # Let inc be the (n-1)-th number in the sorted order of numbers that can be
        # formed using only the bits in free_bits. We form it by writing (n-1) in binary
        # and “translating” its bits. 
        k = 0
        inc = n - 1
        for j in range(r):
            if (inc >> j) & 1:
                k += 1 << free_bits[j]
        
        # The answer is the last element, which is x + k.
        return x + k


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.minEnd(3, 4))  # Expected output: 6, as one optimal array is [4, 5, 6].
    # Example 2:
    print(sol.minEnd(2, 7))  # Expected output: 15, as one optimal array is [7, 15].
    
    # Additional test:
    # Suppose x = 5 (binary 101) and n = 3.
    # Here, free bits are at positions: 1 (2) and then 3 (8) because the lower bit (0) is not free.
    # The two-digit binary numbers we can form (using these bits) in sorted order: 0, 2, 8, 10.
    # The three smallest elements for k are: 0, 2, 8. Thus, the best array is [5, 7, 13].
    # Therefore, the answer should be 13.
    print(sol.minEnd(3, 5))  # Expected output: 13
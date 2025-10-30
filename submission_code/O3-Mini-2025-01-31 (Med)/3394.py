class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Explanation:
        # We want an increasing sequence nums[0..n-1] with nums[i] > 0 where:
        #   1. For every i, the bits that are 1 in x must appear in every nums[i].
        #   2. The bitwise AND over the entire sequence equals exactly x.
        #
        # Write each element as:
        #      nums[i] = x + a[i]
        # where a[i] is a nonnegative integer that only uses bits from positions where x is 0.
        # The reason is that doing x OR a[i] (remember x and a[i] have no overlapping bits)
        # gives x + a[i]. And then:
        #      nums[0] & nums[1] & ... & nums[n-1] = x + (a[0] & a[1] & ... & a[n-1])
        # In order for the overall AND to be exactly x, we need:
        #      a[0] & a[1] & ... & a[n-1] = 0.
        #
        # To minimize the maximum element (nums[n-1]), it is best to choose the sequence so that:
        #     a[0] = 0
        # and then choose a[1], a[2], …, a[n-1] as the (n-1) smallest positive integers 
        # (in increasing order) that can be formed using ONLY the bits in positions where x is 0.
        #
        # Observe that if we list all nonnegative integers that use only allowed bits (zero in x),
        # they are in one-to-one correspondence with the natural numbers via a mapping:
        #    Let free_bits be the sorted list (in increasing order) of bit positions j where (x >> j) & 1 == 0.
        #    Then for any nonnegative integer Y, define:
        #         f(Y) = sum_{i>=0} ( (Y >> i) & 1 ) * (1 << free_bits[i])
        # The sequence of values f(Y) (with Y = 0,1,2,...) is strictly increasing.
        # If we let a[i] = f(i) for i=0 to n-1, we immediately have a[0] = 0,
        # and because a[0] is 0, for each bit position (not in x) even if all other a[i] have that bit, the overall AND will be 0.
        # Therefore, the overall AND becomes x.
        # And the last element will be: nums[n-1] = x + f(n-1).
        #
        # Thus the problem reduces to computing f(n - 1).
        #
        # We do this by iterating over bit positions j (starting from 0 upwards). For each j:
        #   - If x has a 0 in bit j (allowed to use), then take the next bit of Y = n - 1's binary expansion.
        #   - If that bit is 1, add 2^j to f(n-1).
        #   - Continue until we have exhausted all bits of Y (i.e. Y becomes 0).
        #
        # Finally, our answer is x + f(n-1).
        #
        # Special case: When n == 1, the only element in the sequence can be x,
        # and the AND is trivially x.
        if n == 1:
            return x
        
        y = n - 1  # we want to compute f(y)
        add = 0
        bit = 0
        # Process bits in increasing order. We look at bit positions j (0,1,2,…).
        # For every position that x does NOT have (i.e. (x >> bit) & 1 == 0), we look at
        # the least significant bit of y:
        while y > 0:
            if (x >> bit) & 1 == 0:
                if y & 1:
                    add |= (1 << bit)
                y //= 2  # shift y right one position (consume one bit)
            bit += 1
        return x + add


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    n1, x1 = 3, 4
    # Expected output: 6  (One possible sequence: [4,5,6])
    print(sol.minEnd(n1, x1))  # Output: 6

    # Example 2:
    n2, x2 = 2, 7
    # Expected output: 15 (One possible sequence: [7,15])
    print(sol.minEnd(n2, x2))  # Output: 15
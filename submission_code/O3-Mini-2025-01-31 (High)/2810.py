class Solution:
    def minCost(self, nums: list[int], x: int) -> int:
        # Explanation:
        # We are given an array of n “prices” for chocolates of n different types.
        # Initially the chocolate of type i costs nums[i]. In one “operation” (rotation)
        # costing x we relabel the types: every chocolate of type i becomes type ((i+1) mod n).
        #
        # In other words, if you “delay” buying a chocolate (of some needed type) until after r rotations,
        # then the price available for that type will come from a different index.
        # In fact, after r rotations, the chocolate available for type i originally came from index (i–r mod n).
        #
        # HOWEVER – the twist is that the operations are done globally.
        # That is, if you want to “improve” the price you pay for your collection,
        # you may choose to perform some rotations (each costing x) and then buy some types with
        # the (possibly) lowered price.
        #
        # When you perform R rotations overall (in a “smart sequence” of interleaving buys and rotations)
        # you are allowed, for each type i, to “buy” at an earlier round (with r < R) or wait until later.
        #
        # This means that in an optimal strategy the extra cost paid for rotations is R*x (where R is
        # the total number of operations you eventually do) and you are free, for every type i,
        # to choose any option with a delay r (0 <= r <= R). For a fixed type i, the candidate prices are:
        #   nums[i], nums[(i-1) mod n], nums[(i-2) mod n], …, nums[(i-R) mod n].
        #
        # Hence, if we define:
        #    m_R[i] = min{ nums[i], nums[(i-1) mod n], …, nums[(i-R) mod n] }
        #
        # then if you plan to use R rotations in total the overall cost will be:
        #    cost = sum(m_R[i] for i in 0..n-1) + R*x.
        #
        # Our answer is the minimum of cost over R = 0,1,…,n-1.
        #
        # We can compute these m_R[i] values incrementally.
        n = len(nums)
        # Let m[i] denote the best (lowest) price available at type i from some candidate
        # in the “window” of delay choices. Initially, with R = 0, m[i] = nums[i].
        m = nums[:]  
        best = sum(m)  # no rotation performed, R = 0
        # For additional rotations we update m[i] by considering an extra candidate.
        for R in range(1, n):
            for i in range(n):
                # The extra candidate for type i is the chocolate coming from index (i - R) mod n.
                candidate = nums[(i - R) % n]
                if candidate < m[i]:
                    m[i] = candidate
            total = sum(m) + R * x
            if total < best:
                best = total
        return best


# ---------- Testing the solution with some examples ----------

if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    # Explanation:
    #   With nums = [20,1,15] and x = 5 one optimal strategy is:
    #   • Buy type1 (cost = 1) immediately.
    #   • Do one rotation (cost = 5) so that the chocolate with price 1 shifts into (for example) type2.
    #   • Buy type2 (cost = 1).
    #   • Do one more rotation (cost = 5) so that the cheap chocolate (price 1) appears as type0.
    #   • Buy type0 (cost = 1).
    #   Total cost = 1 + 5 + 1 + 5 + 1 = 13.
    print(sol.minCost([20, 1, 15], 5))  # Expected output: 13

    # Example 2:
    # Here performing rotations is too expensive relative to the chocolate prices.
    # So it is best to buy each type “as is”: 1 + 2 + 3 = 6.
    print(sol.minCost([1, 2, 3], 4))    # Expected output: 6
class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        counts = [0]*31  # counts of 2^i

        for num in nums:
            e = num.bit_length() - 1  # since nums[i] is power of two
            counts[e] +=1

        required = [(target >> i) &1 for i in range(31)]

        ans = 0
        for i in range(31):
            # First, ensure counts[i] are enough for required[i]
            if counts[i] >= required[i]:
                # We can pass excess counts to next bit
                pass
            else:
                # We need to make up deficit counts
                deficit = required[i] - counts[i]
                # Find higher bits to split
                j = i + 1
                while j < 31 and counts[j] == 0:
                    j +=1
                if j == 31:
                    return -1  # impossible to satisfy target
                # Split counts[j] down to counts[i]
                for k in range(j, i, -1):
                    counts[k] -=1
                    counts[k -1] +=2
                    ans +=1  # one operation per split
                counts[i] +=1  # Now counts[i] has at least required[i]

            # Pass excess counts to next bit if any
            excess = counts[i] - required[i]
            if i +1 <31:
                counts[i+1] += excess //2

        return ans
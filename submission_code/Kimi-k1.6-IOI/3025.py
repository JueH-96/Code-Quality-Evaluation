class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        count = [0] * 31  # bits 0 to 30
        for num in nums:
            # Calculate the exponent for the power of two
            exponent = num.bit_length() - 1
            count[exponent] += 1
        
        total_sum = sum(nums)
        if total_sum < target:
            return -1
        
        ans = 0
        
        for i in range(31):
            # Determine if the current bit is required in the target
            required = (target >> i) & 1
            if required:
                if count[i] == 0:
                    # Find the smallest higher bit with available elements
                    j = i + 1
                    while j < 31 and count[j] == 0:
                        j += 1
                    if j == 31:
                        # This should not happen as sum >= target
                        return -1
                    ans += (j - i)
                    count[j] -= 1
                    count[i] += (1 << (j - i))  # Add 2^(j-i) elements to current bit
                # Use one element for the required bit
                count[i] -= 1
            # Carry over surplus elements to the next higher bit
            surplus = count[i]
            count[i] = surplus % 2
            if i + 1 < 31:
                count[i + 1] += surplus // 2
        
        return ans
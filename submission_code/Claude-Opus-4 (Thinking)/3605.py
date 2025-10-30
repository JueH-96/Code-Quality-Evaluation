class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for t in nums:
            if t % 2 == 0:  # t ends with 0 in binary
                ans.append(-1)
            else:
                # t has some trailing 1s. Find 2^k where k is the number of trailing 1s
                # t+1 has k trailing 0s, so its lowbit is 2^k
                power_of_2_k = (t + 1) & -(t + 1)
                # We want to subtract 2^{k-1} from t
                x = t - (power_of_2_k >> 1)
                ans.append(x)
        return ans
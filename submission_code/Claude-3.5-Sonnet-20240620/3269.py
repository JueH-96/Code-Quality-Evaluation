class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        def kmp(text, pattern):
            lps = [0] * len(pattern)
            length = 0
            i = 1
            while i < len(pattern):
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                elif length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
            
            count = 0
            i = j = 0
            while i < len(text):
                if pattern[j] == text[i]:
                    i += 1
                    j += 1
                if j == len(pattern):
                    count += 1
                    j = lps[j - 1]
                elif i < len(text) and pattern[j] != text[i]:
                    if j != 0:
                        j = lps[j - 1]
                    else:
                        i += 1
            return count

        encoded = []
        for i in range(len(nums) - 1):
            if nums[i+1] > nums[i]:
                encoded.append(1)
            elif nums[i+1] == nums[i]:
                encoded.append(0)
            else:
                encoded.append(-1)
        
        return kmp(encoded, pattern)
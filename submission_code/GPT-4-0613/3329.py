class Solution:
    def longestCommonPrefix(self, arr1, arr2):
        arr1 = [str(i) for i in arr1]
        arr2 = [str(i) for i in arr2]
        max_len = 0
        for i in arr1:
            for j in arr2:
                common_prefix = 0
                min_len = min(len(i), len(j))
                for k in range(min_len):
                    if i[k] == j[k]:
                        common_prefix += 1
                    else:
                        break
                max_len = max(max_len, common_prefix)
        return max_len
class Solution:
    def countPairs(self, nums: List[int]) -> int:
        from collections import defaultdict

        def get_almost_equal_forms(num):
            s = str(num)
            forms = set()
            forms.add(s)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    # Swap characters at positions i and j
                    new_s = list(s)
                    new_s[i], new_s[j] = new_s[j], new_s[i]
                    forms.add(''.join(new_s))
            return forms

        # Dictionary to store the forms of each number
        forms_dict = defaultdict(list)

        for i, num in enumerate(nums):
            forms = get_almost_equal_forms(num)
            for form in forms:
                forms_dict[form].append(i)

        count = 0
        for indices in forms_dict.values():
            n = len(indices)
            if n > 1:
                # Number of ways to choose 2 indices from n indices
                count += n * (n - 1) // 2

        return count
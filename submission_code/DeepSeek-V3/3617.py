class Solution:
    def possibleStringCount(self, word: str) -> int:
        n = len(word)
        if n == 1:
            return 1
        count = 0
        for i in range(n):
            if i == 0:
                if word[i] != word[i+1]:
                    count += 1
            elif i == n-1:
                if word[i] != word[i-1]:
                    count += 1
            else:
                if word[i] != word[i-1] and word[i] != word[i+1]:
                    count += 1
        # Now, consider the case where a single character is repeated
        # We need to find all positions where a character is repeated at least twice
        # and count the number of ways to reduce it to one
        # For example, for "aaaa", we can reduce any of the 'a's to one
        # So, for each group of consecutive same characters, if the length is >= 2, we add the length - 1
        # But since we can only have at most one such reduction, we need to consider all possible single reductions
        # So, for each group of consecutive same characters of length k >= 2, we add (k-1) to the count
        # But since we can only have one such reduction, the total is the sum of all (k-1) for each group
        # Wait, no. For each group of length k >= 2, we can choose any one of the k-1 positions to reduce
        # So for each group, we add (k-1) to the count
        # But since we can only have one such reduction, the total is the sum of all (k-1) for each group
        # So, we need to find all groups of consecutive same characters and sum their (length - 1)
        # Then, the total count is the initial count plus the sum of (length - 1) for each group
        # So, let's compute that
        group_counts = []
        current_char = word[0]
        current_length = 1
        for i in range(1, n):
            if word[i] == current_char:
                current_length += 1
            else:
                if current_length >= 2:
                    group_counts.append(current_length)
                current_char = word[i]
                current_length = 1
        if current_length >= 2:
            group_counts.append(current_length)
        # Now, sum all (length - 1) for each group
        total_reductions = sum([length - 1 for length in group_counts])
        # The total possible original strings is the initial count plus the total_reductions
        # Because for each reduction, we have a new possible original string
        return count + total_reductions
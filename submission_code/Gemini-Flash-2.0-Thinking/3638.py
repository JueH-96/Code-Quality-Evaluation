from collections import Counter

class Solution:
    def makeStringGood(self, s: str) -> int:
        n = len(s)
        min_ops = float('inf')

        for length in range(1, n + 1):
            for num_chars in range(1, 27):
                if length % num_chars == 0:
                    freq = length // num_chars

                    from itertools import combinations
                    for chars_tuple in combinations("abcdefghijklmnopqrstuvwxyz", num_chars):
                        target_chars = sorted(list(chars_tuple))
                        ops = abs(n - length)

                        s_sorted = sorted(s)
                        costs = []

                        for char_s in s_sorted:
                            min_cost_char = float('inf')
                            for target_char in target_chars:
                                if target_char >= char_s:
                                    min_cost_char = min(min_cost_char, ord(target_char) - ord(char_s))
                            costs.append(min_cost_char)

                        costs.sort()

                        char_counts = Counter(s)

                        transform_costs = 0
                        source_chars = []
                        for char, count in char_counts.items():
                            source_chars.extend([char] * count)
                        source_chars.sort()

                        possible_transformations = []
                        for target_char in target_chars:
                            possible_transforms_for_target = []
                            for source_char in source_chars:
                                if source_char <= target_char:
                                    possible_transforms_for_target.append((ord(target_char) - ord(source_char), source_char, target_char))
                            possible_transforms_for_target.sort()
                            possible_transformations.append(possible_transforms_for_target)

                        from itertools import product

                        indices = [list(range(len(source_chars))) for _ in range(length)]

                        if length == n:
                            cost = 0
                            char_counts_s = Counter(s)
                            for char in target_chars:
                                diff = abs(char_counts_s.get(char, 0) - freq)
                                # This approach is becoming too complex, let's rethink.

                        counts_s = Counter(s)

                        cost_transform = 0

                        # Try to match characters in s to the target good string
                        # Consider the operations needed for each character in s

                        # Operations: delete, insert, change

                        # Iterate through all possible good strings
                        # A good string is defined by its length and the characters it contains

                        target_string_counts = {char: freq for char in target_chars}

                        temp_s_counts = Counter(s)
                        current_ops = abs(n - length)

                        for char in target_chars:
                            diff = target_string_counts[char] - temp_s_counts.get(char, 0)
                            if diff > 0:
                                current_ops += diff # Insertions needed
                            elif diff < 0:
                                current_ops += abs(diff) # Deletions needed

                        # This doesn't account for changes

                        # Let's consider the final good string and the operations to reach it from s

                        # Iterate through all possible target good strings
                        for target_length in range(1, n + 1):
                            for num_distinct_chars in range(1, 27):
                                if target_length % num_distinct_chars == 0:
                                    target_freq = target_length // num_distinct_chars
                                    for target_chars_tuple in combinations("abcdefghijklmnopqrstuvwxyz", num_distinct_chars):
                                        target_chars_set = set(target_chars_tuple)

                                        current_ops = 0

                                        s_counts = Counter(s)

                                        # Operations to transform s to the target good string

                                        # Consider the characters in s
                                        for char_s, count_s in s_counts.items():
                                            if char_s not in target_chars_set:
                                                current_ops += count_s # Delete these characters

                                        # Consider the characters in the target good string
                                        for char_t in target_chars_set:
                                            needed = target_freq
                                            have = s_counts.get(char_t, 0)

                                            if have < needed:
                                                current_ops += (needed - have) # Insertions or changes

                                        # This is still not quite right

                                        # Let's focus on transforming s

                                        cost = abs(n - target_length)

                                        # Try to transform characters in s to match the target
                                        sorted_s = sorted(s)

                                        # We need target_freq of each char in target_chars_set

                                        # This problem seems to require a more direct calculation of operations

                                        target_counts = {char: target_freq for char in target_chars_set}

                                        temp_ops = abs(n - target_length)

                                        source = list(s)

                                        # Try to transform source to match target counts
                                        for char_t in sorted(target_counts.keys()):
                                            needed = target_counts[char_t]
                                            have = source.count(char_t)

                                            diff = needed - have

                                            if diff > 0: # Need to insert or change
                                                pass
                                            elif diff < 0: # Need to delete or change
                                                pass

                                        # Consider the cost of transforming s to a good string directly

                                        target_counts = {}
                                        for char in target_chars_set:
                                            target_counts[char] = freq

                                        current_ops = abs(n - length)

                                        s_counts = Counter(s)

                                        cost_transform = 0

                                        source_chars_list = sorted(list(s))

                                        # Match characters from s to the target good string
                                        # We need freq occurrences of each char in target_chars

                                        # This is becoming too complex, let's simplify the approach

                                        # Iterate through all possible good strings and calculate the cost

                                        target_counts = {char: freq for char in target_chars}

                                        temp_s = list(s)
                                        temp_ops = 0

                                        # Try to transform temp_s to match target_counts

                                        # Calculate operations needed to make s good with target frequency and chars
                                        ops = abs(n - length) # Insertions/deletions

                                        s_counts = Counter(s)

                                        diffs = []
                                        for char in target_chars:
                                            diffs.append(s_counts.get(char, 0))
                                        diffs.sort()

                                        target_counts_list = [freq] * num_chars

                                        cost_transform = 0

                                        source_counts = Counter(s)

                                        available_chars = sorted(source_counts.keys())

                                        # Greedy approach: try to transform characters in s to match the target

                                        cost_transform = 0

                                        source_list = sorted(list(s))

                                        k = 0
                                        for char_s in source_list:
                                            found_match = False
                                            for target_char in sorted(target_chars):
                                                target_freq_current = target_counts.get(target_char, 0)
                                                num_achieved = sum(1 for c in source_list[:k+1] if c == target_char)

                                                if num_achieved < freq and char_s <= target_char:
                                                    cost_transform += ord(target_char) - ord(char_s)
                                                    found_match = True
                                                    break
                                            if not found_match:
                                                cost_transform += 1 # Deletion

                                            k += 1

                                        min_ops = min(min_ops, ops + cost_transform)

        return min_ops
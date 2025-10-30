class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from bisect import bisect_left, bisect_right, insort_left

        counts = {}  # ID -> current count
        counts_counts = {}  # count -> set of IDs with that count
        counts_counts_keys = []  # sorted list of counts

        ans = []
        for i in range(len(nums)):
            ID = nums[i]
            delta = freq[i]

            c_old = counts.get(ID, 0)
            c_new = c_old + delta
            counts[ID] = c_new

            # Remove ID from old count in counts_counts
            if c_old != 0:
                counts_counts[c_old].remove(ID)
                if not counts_counts[c_old]:
                    # Remove count c_old from counts_counts and counts_counts_keys
                    del counts_counts[c_old]
                    idx = bisect_left(counts_counts_keys, c_old)
                    if idx < len(counts_counts_keys) and counts_counts_keys[idx] == c_old:
                        counts_counts_keys.pop(idx)
            # Add ID to new count in counts_counts
            if c_new in counts_counts:
                counts_counts[c_new].add(ID)
            else:
                counts_counts[c_new] = set([ID])
                # Insert c_new into counts_counts_keys
                insort_left(counts_counts_keys, c_new)

            # Update max_count
            if counts_counts_keys:
                max_count = counts_counts_keys[-1]
            else:
                max_count = 0

            ans.append(max_count)

        return ans
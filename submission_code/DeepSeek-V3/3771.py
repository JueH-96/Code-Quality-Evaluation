class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k == 0:
            return True
        
        # Create a dictionary to store the first and last occurrence of each character
        first_occurrence = {}
        last_occurrence = {}
        
        for idx, char in enumerate(s):
            if char not in first_occurrence:
                first_occurrence[char] = idx
            last_occurrence[char] = idx
        
        # Collect all intervals [first, last] for each character
        intervals = []
        for char in first_occurrence:
            start = first_occurrence[char]
            end = last_occurrence[char]
            intervals.append((start, end))
        
        # Merge intervals to find the minimal set of intervals covering all characters' ranges
        if not intervals:
            return False
        
        # Sort intervals based on start time
        intervals.sort()
        merged = []
        current_start, current_end = intervals[0]
        
        for start, end in intervals[1:]:
            if start <= current_end:
                # Overlapping or adjacent intervals, merge them
                current_start = min(current_start, start)
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        
        # Now, within each merged interval, find the maximal number of non-overlapping special substrings
        total = 0
        for start, end in merged:
            if start == 0 and end == len(s) - 1:
                # The entire string is a candidate, but special substrings cannot be the entire string
                # So we need to find substrings within this interval
                # We need to find intervals where all characters in the substring have their first and last within the substring
                # So we can split into intervals where certain characters are confined
                # This part is tricky; perhaps we can look for minimal such intervals
                # We can proceed by finding all characters that have their start and end within some sub-interval
                # Then, these are candidates for special substrings
                # We can model this as finding all such minimal intervals
                # Initialize variables
                sub_intervals = []
                # Rebuild intervals for characters within this merged interval
                sub_char_intervals = []
                for char in first_occurrence:
                    char_start = first_occurrence[char]
                    char_end = last_occurrence[char]
                    if char_start >= start and char_end <= end:
                        sub_char_intervals.append((char_start, char_end))
                # Now, process these intervals to find non-overlapping ones
                if not sub_char_intervals:
                    continue
                sub_char_intervals.sort()
                # The problem reduces to finding maximal number of non-overlapping intervals
                # But each interval must be a special substring: i.e., all characters in it have their first and last within it
                # So, we need to find intervals that are themselves special
                # So, for each possible interval, check if it's special
                # But this is O(n^2), which is not feasible for large n
                # Alternative approach: find all minimal intervals that are special
                # But how?
                pass
                # This part requires a different approach
                # We can use a greedy algorithm to pick the earliest ending intervals
                # But first, we need to find all possible intervals that are special
                # So, for each character, the interval [first, last] is a candidate, but it may include other characters
                # So, we need to find all intervals where for every character in the interval, their first and last are within the interval
                # So, for each possible start and end, check if all characters in s[start..end] have their first and last in [start..end]
                # But this is O(n^2), which is not feasible for large n
                # Hence, we need an optimized approach
                # Let's think differently: the special substrings are those where the set of characters in the substring is disjoint from the set of characters outside the substring
                # So, for a substring s[l..r], the set of characters in s[l..r] must not appear in s[0..l-1] or s[r+1..n-1]
                # So, the approach is to find all such substrings
                # Then, the problem reduces to selecting k non-overlapping such substrings
                # To find all such substrings efficiently, we can proceed as follows:
                # For each position l, find the minimal r such that all characters in s[l..r] do not appear before l or after r
                # This can be done using the first and last occurrence maps
                n = len(s)
                l = 0
                special_intervals = []
                while l < n:
                    # Find the minimal r >= l such that for all characters in s[l..r], their first >= l and last <= r
                    current_chars = set()
                    r_candidate = l
                    max_last = -1
                    valid = True
                    # We need to expand r until all characters in s[l..r] have their last <= r and first >= l
                    temp_r = l
                    initial_max_last = last_occurrence[s[l]]
                    if first_occurrence[s[l]] < l:
                        l += 1
                        continue
                    current_max_last = initial_max_last
                    temp_r = l
                    while temp_r <= current_max_last:
                        char = s[temp_r]
                        if first_occurrence[char] < l or last_occurrence[char] > current_max_last:
                            current_max_last = max(current_max_last, last_occurrence[char])
                        temp_r += 1
                    # Now, check if the interval s[l..current_max_last] is valid
                    # i.e., all characters in it have first >= l and last <= current_max_last
                    valid_interval = True
                    for i in range(l, current_max_last + 1):
                        char = s[i]
                        if first_occurrence[char] < l or last_occurrence[char] > current_max_last:
                            valid_interval = False
                            break
                    if valid_interval and current_max_last < n and l <= current_max_last and (l != 0 or current_max_last != n - 1):
                        special_intervals.append((l, current_max_last))
                        l = current_max_last + 1
                    else:
                        l += 1
                # Now, select the maximal number of non-overlapping intervals from special_intervals
                if not special_intervals:
                    continue
                # Sort intervals by end time
                special_intervals.sort(key=lambda x: x[1])
                selected = []
                last_end = -1
                count = 0
                for interval in special_intervals:
                    if interval[0] > last_end:
                        selected.append(interval)
                        last_end = interval[1]
                        count += 1
                total += count
            else:
                # This interval is a special substring because it's not the entire string
                total += 1
        
        return total >= k
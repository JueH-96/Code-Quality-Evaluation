from collections import defaultdict

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        # Initialize the list to store the lengths of alternating groups
        group_lengths = []
        # To handle circular nature, we can treat the array as linear and handle the wrap-around
        # by checking the first and last elements when necessary.
        
        # First, find all the alternating groups and their lengths
        # We can iterate through the array and identify the groups
        # Since it's circular, we need to handle the wrap-around
        # We can treat the array as linear and then handle the wrap-around separately
        
        # Initialize the current group length
        current_length = 1
        # Initialize the list to store the lengths of all groups
        lengths = []
        # Iterate through the array
        for i in range(1, n):
            if colors[i] != colors[i-1]:
                current_length += 1
            else:
                lengths.append(current_length)
                current_length = 1
        # Handle the wrap-around
        if colors[0] != colors[-1]:
            # If the first and last are different, we need to merge the first and last groups
            # Find the first group length
            first_length = 1
            i = 1
            while i < n and colors[i] != colors[i-1]:
                first_length += 1
                i += 1
            # Find the last group length
            last_length = 1
            i = n - 2
            while i >= 0 and colors[i] != colors[i+1]:
                last_length += 1
                i -= 1
            # If the first and last groups are part of the same alternating sequence
            if first_length + last_length <= n:
                lengths.append(first_length + last_length)
            else:
                lengths.append(first_length)
                lengths.append(last_length)
        else:
            lengths.append(current_length)
        
        # Now, we need to count the number of groups of each length
        # We can use a dictionary to count the frequency of each length
        length_counts = defaultdict(int)
        for length in lengths:
            length_counts[length] += 1
        
        # Now, process the queries
        result = []
        for query in queries:
            if query[0] == 1:
                size = query[1]
                result.append(length_counts.get(size, 0))
            else:
                # Update the colors array
                index = query[1]
                new_color = query[2]
                old_color = colors[index]
                if old_color == new_color:
                    # No change, skip
                    continue
                # Update the color
                colors[index] = new_color
                # Now, we need to recompute the lengths of the groups
                # This is a bit tricky, but for the sake of time, we will recompute all lengths
                # This is not efficient for large n, but given the constraints, it should work
                lengths = []
                current_length = 1
                for i in range(1, n):
                    if colors[i] != colors[i-1]:
                        current_length += 1
                    else:
                        lengths.append(current_length)
                        current_length = 1
                # Handle the wrap-around
                if colors[0] != colors[-1]:
                    first_length = 1
                    i = 1
                    while i < n and colors[i] != colors[i-1]:
                        first_length += 1
                        i += 1
                    last_length = 1
                    i = n - 2
                    while i >= 0 and colors[i] != colors[i+1]:
                        last_length += 1
                        i -= 1
                    if first_length + last_length <= n:
                        lengths.append(first_length + last_length)
                    else:
                        lengths.append(first_length)
                        lengths.append(last_length)
                else:
                    lengths.append(current_length)
                # Recompute the length_counts
                length_counts = defaultdict(int)
                for length in lengths:
                    length_counts[length] += 1
        return result
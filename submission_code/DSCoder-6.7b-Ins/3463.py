class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        groups = 0
        n = len(colors)

        # Initialize the start of a new group
        i = 0

        while i < n:
            # Find the end of the current group
            while i < n - 1 and colors[i] == colors[i + 1]:
                i += 1

            # Find the start of the next group
            while i < n - 1 and colors[i] != colors[i + 1]:
                i += 1

            # Increment the group count
            groups += 1

        return groups
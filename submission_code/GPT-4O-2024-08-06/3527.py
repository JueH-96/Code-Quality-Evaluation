class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def count_alternating_groups(size):
            n = len(colors)
            count = 0
            # Check all possible starting points for a group of size `size`
            for start in range(n):
                valid = True
                for i in range(size):
                    if colors[(start + i) % n] == colors[(start + i + 1) % n]:
                        valid = False
                        break
                if valid:
                    count += 1
            return count

        results = []
        for query in queries:
            if query[0] == 1:
                # Count alternating groups of the given size
                size = query[1]
                results.append(count_alternating_groups(size))
            elif query[0] == 2:
                # Update the color at the given index
                index, color = query[1], query[2]
                colors[index] = color

        return results
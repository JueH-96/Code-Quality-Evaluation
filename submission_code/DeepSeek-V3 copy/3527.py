class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        # Precompute all possible alternating groups and their sizes
        # Since the tiles are circular, we need to handle the wrap-around
        # We can represent the groups as tuples of indices
        # However, precomputing all groups is computationally expensive for large n
        # Instead, we can compute the count of alternating groups of a given size on the fly
        
        # To handle the circular nature, we can treat the array as linear and then handle the wrap-around
        # For example, for a group of size k, we need to check the sequence from i to i+k-1, wrapping around if necessary
        
        # We will process each query one by one
        answer = []
        for query in queries:
            if query[0] == 1:
                size = query[1]
                count = 0
                for i in range(n):
                    # Check if the sequence from i to i+size-1 is alternating
                    # Handle wrap-around
                    is_alternating = True
                    for j in range(size):
                        idx = (i + j) % n
                        next_idx = (i + j + 1) % n
                        if j < size - 1 and colors[idx] == colors[next_idx]:
                            is_alternating = False
                            break
                    if is_alternating:
                        count += 1
                answer.append(count)
            elif query[0] == 2:
                index = query[1]
                color = query[2]
                colors[index] = color
        return answer
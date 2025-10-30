import heapq

class Solution:
    def minOrAfterOperations(self, nums):
        import sys
        sys.setrecursionlimit(1 << 25)
        n = len(nums)
        if n == 1 or len(set(nums)) == 1:
            return nums[0]
        k = self.k
        class Node:
            def __init__(self, val):
                self.val = val
                self.left = None
                self.right = None
                self.removed = False
        nodes = [Node(val) for val in nums]
        for i in range(len(nodes) -1):
            nodes[i].right = nodes[i+1]
            nodes[i+1].left = nodes[i]
        heap = []
        index = 0
        for node in nodes[:-1]:
            merged_val = node.val & node.right.val
            heapq.heappush(heap, (merged_val, index, node))
            index +=1
        count = 0
        removed = set()
        while heap and count < k:
            val, idx, node = heapq.heappop(heap)
            if node.removed or node.right.removed:
                continue
            node.val = val
            node.removed = False
            node.right.removed = True
            # Update links
            next_node = node.right.right
            node.right = next_node
            if next_node:
                next_node.left = node
                # Push new pair to heap
                merged_val = node.val & next_node.val
                heapq.heappush(heap, (merged_val, index, node))
                index +=1
            if node.left:
                prev_node = node.left
                # Push new pair to heap
                merged_val = prev_node.val & node.val
                heapq.heappush(heap, (merged_val, index, prev_node))
                index +=1
            count +=1
        # Collect remaining values
        res = []
        node = None
        for n in nodes:
            if not n.removed:
                node = n
                break
        while node:
            res.append(node.val)
            node = node.right
        # Compute OR of remaining elements
        ans = 0
        for val in res:
            ans |= val
        return ans

# Wrapper function to set k value
def minOrAfterOperationsWrapper(nums, k):
    solution = Solution()
    solution.k = k
    return solution.minOrAfterOperations(nums)

# For online judge, comment out the function above and use this class directly.
import math
import random
from typing import List

# We'll use a Treap (a randomized balanced BST) that supports insertion and 
# queries for the lower_bound (smallest key >= target) and the predecessor 
# (greatest key < target). In our solution, as we move through the array,
# the treap will hold all previously‐seen elements that are at least x indices before
# the current index. Then for the current element, we query for the closest value 
# (both above and below) to update our answer.

class Node:
    __slots__ = ('key', 'priority', 'count', 'left', 'right')
    def __init__(self, key: int):
        self.key = key
        self.priority = random.random()  # Use a random float as priority.
        self.count = 1   # To handle duplicate values.
        self.left = None
        self.right = None

def rotate_right(root: Node) -> Node:
    new_root = root.left
    root.left = new_root.right
    new_root.right = root
    return new_root

def rotate_left(root: Node) -> Node:
    new_root = root.right
    root.right = new_root.left
    new_root.left = root
    return new_root

def treap_insert(root: Node, key: int) -> Node:
    if root is None:
        return Node(key)
    if key == root.key:
        root.count += 1
        return root
    elif key < root.key:
        root.left = treap_insert(root.left, key)
        if root.left.priority < root.priority:
            root = rotate_right(root)
    else:  # key > root.key
        root.right = treap_insert(root.right, key)
        if root.right.priority < root.priority:
            root = rotate_left(root)
    return root

def treap_lower_bound(root: Node, key: int):
    # Returns the node with the smallest key >= given key, or None if not found.
    if root is None:
        return None
    if root.key >= key:
        candidate = treap_lower_bound(root.left, key)
        return candidate if candidate is not None else root
    else:
        return treap_lower_bound(root.right, key)

def treap_predecessor(root: Node, key: int):
    # Returns the node with the greatest key < given key, or None if not found.
    if root is None:
        return None
    if root.key < key:
        candidate = treap_predecessor(root.right, key)
        return candidate if candidate is not None else root
    else:
        return treap_predecessor(root.left, key)

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if n < 2:
            return 0  # Not enough numbers to form a valid pair.
        
        # When x is 0 the condition "abs(i - j) >= 0" holds for any two distinct indices.
        # In that case the answer is the minimum absolute difference between any two numbers.
        if x == 0:
            nums_sorted = sorted(nums)
            ans = math.inf
            for i in range(1, n):
                ans = min(ans, nums_sorted[i] - nums_sorted[i-1])
            return ans

        ans = math.inf
        root = None  # Our treap will store eligible numbers (values from indices at least x back)
        for j in range(n):
            # For index j, all indices i such that i <= j-x have been processed.
            if j - x >= 0:
                root = treap_insert(root, nums[j - x])
            if root is not None:
                # Query treap for candidate with value >= nums[j].
                cand = treap_lower_bound(root, nums[j])
                if cand is not None:
                    ans = min(ans, abs(nums[j] - cand.key))
                # Query treap for candidate with value < nums[j].
                cand2 = treap_predecessor(root, nums[j])
                if cand2 is not None:
                    ans = min(ans, abs(nums[j] - cand2.key))
        return ans

# For quick testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1: indices 0 and 3 are at least 2 apart and |4-4| = 0.
    print(sol.minAbsoluteDifference([4, 3, 2, 4], 2))    # Expected output: 0
    # Example 2: valid pair is nums[1]=3 and nums[2]=2 with difference 1.
    print(sol.minAbsoluteDifference([5, 3, 2, 10, 15], 1)) # Expected output: 1
    # Example 3: valid pair is nums[0]=1 and nums[3]=4 with difference 3.
    print(sol.minAbsoluteDifference([1, 2, 3, 4], 3))      # Expected output: 3
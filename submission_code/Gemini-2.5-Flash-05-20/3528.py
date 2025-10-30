import collections

# Define a Line class to represent y = mx + c
class Line:
    def __init__(self, m, c):
        self.m = m
        self.c = c
    
    # Evaluate the line at a given x-coordinate
    def eval(self, x):
        return self.m * x + self.c

# Node class for the Li Chao Tree
class _Node:
    def __init__(self, line):
        self.left = None
        self.right = None
        self.line = line # The line stored in this node

class LiChaoTree:
    # A default line that returns a very small value (negative infinity)
    # Used for segments that don't have any actual lines inserted yet.
    DEFAULT_LINE = Line(0, -float('inf')) 

    def __init__(self, x_min, x_max):
        # The range of x-coordinates the tree will cover [x_min, x_max]
        self.x_min = x_min
        self.x_max = x_max
        self.root = None # Root node of the Li Chao tree

    # Adds a new line (m, c) to the Li Chao Tree
    def add_line(self, m, c):
        new_line = Line(m, c)
        # If the tree is empty, the new line becomes the root's line
        if self.root is None:
            self.root = _Node(new_line)
            return

        # Start the recursive insertion from the root
        self.root = self._add_line_recursive(self.root, new_line, self.x_min, self.x_max)

    # Recursive helper function to add a line
    def _add_line_recursive(self, node, new_line, low, high):
        # If no node exists for this segment, create one and store the new_line
        if node is None:
            return _Node(new_line)

        # Get the line currently stored in the node
        line_in_node = node.line
        
        # Base case: if the segment is a single point, decide which line is better
        # and store it in the node.
        if low == high:
            if new_line.eval(low) > line_in_node.eval(low):
                node.line = new_line
            return node

        mid = low + (high - low) // 2 # Calculate the midpoint of the current segment

        # Evaluate both lines at the midpoint to decide which one is "better"
        val_new_mid = new_line.eval(mid)
        val_node_mid = line_in_node.eval(mid)

        # If the new line is better at the midpoint, swap it with the line currently in the node.
        # The line that was worse at midpoint (original `line_in_node` or the original `new_line`)
        # is now referred to as `new_line` and needs to be pushed to a child.
        if val_new_mid > val_node_mid:
            node.line, new_line = new_line, line_in_node # Swap references

        # Now `new_line` is the line that lost at `mid`. We need to determine which child
        # (`left` or `right`) it might dominate and thus should be inserted into.
        # This is determined by comparing their values at the lower endpoint (`low`).
        
        # If `new_line` is better at the lower endpoint than `node.line` (the current best),
        # it implies `new_line` is less steep and its intersection point with `node.line` is to the right of `low`.
        # Therefore, `new_line` could potentially be better in the left half of the segment.
        if new_line.eval(low) > node.line.eval(low):
            node.left = self._add_line_recursive(node.left, new_line, low, mid)
        # Otherwise (if `new_line` is better at the higher endpoint, or worse at both ends than `node.line`),
        # it implies `new_line` is steeper (or crosses `node.line` before `mid`).
        # Therefore, `new_line` could potentially be better in the right half.
        else:
            node.right = self._add_line_recursive(node.right, new_line, mid + 1, high)
        
        return node

    # Queries the Li Chao Tree for the maximum y-value at a given x-coordinate
    def query(self, x):
        curr = self.root
        low, high = self.x_min, self.x_max
        max_val = -float('inf') # Initialize max_val to negative infinity

        # Traverse the tree iteratively to find the maximum value at x
        while curr:
            # Evaluate the line stored in the current node at x and update max_val
            max_val = max(max_val, curr.line.eval(x))
            
            # Decide which child segment to explore based on x
            mid = low + (high - low) // 2
            if x <= mid:
                curr = curr.left
                high = mid # Narrow down the range for the next iteration
            else:
                curr = curr.right
                low = mid + 1 # Narrow down the range for the next iteration
        return max_val


class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        
        # dp[i] will store the maximum score to reach index i.
        # Initialize with -infinity, except for dp[0].
        # Using a list to hold dp values.
        dp = [-float('inf')] * n
        dp[0] = 0 # Score to reach index 0 is 0 (no jumps needed)

        # Initialize the Li Chao Tree. It operates on the domain of x-coordinates,
        # which are the array indices from 0 to n-1.
        lct = LiChaoTree(0, n - 1)

        # Add the initial line corresponding to j=0 to the Li Chao Tree.
        # This line will be used when calculating dp[i] for i > 0.
        # For j=0:
        #   slope m = nums[0]
        #   y-intercept c = dp[0] - 0 * nums[0] = 0 - 0 = 0
        lct.add_line(nums[0], 0)

        # Iterate from i=1 to n-1 to compute dp[i]
        for i in range(1, n):
            # Query the Li Chao Tree for the maximum value at x = i.
            # This query effectively finds max(dp[j] - j * nums[j] + i * nums[j]) over j < i.
            # This result is precisely the maximum score to reach index 'i' from any previous index 'j'.
            dp[i] = lct.query(i)
            
            # After computing dp[i], add the line corresponding to index i
            # into the Li Chao Tree. This line will be used for future queries
            # (i.e., for computing dp[k] where k > i).
            # For j=i:
            #   slope m = nums[i]
            #   y-intercept c = dp[i] - i * nums[i]
            lct.add_line(nums[i], dp[i] - i * nums[i])
        
        # The goal is to reach the last index (n-1), so return dp[n-1].
        return dp[n - 1]
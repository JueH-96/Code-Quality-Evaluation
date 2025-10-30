# YOUR CODE HERE
import sys
import collections

# A small optimization for speed in Python competitive programming
input = sys.stdin.readline

# Read N, the number of strings.
# The constraints guarantee N is at least 2, so no need to handle empty input.
try:
    N = int(input())
except (ValueError, IndexError):
    # Handles cases where input might be empty, though not expected by problem constraints
    N = 0

# We will use a Trie data structure to efficiently process the prefixes of all strings.
# A node in the Trie is represented by a list: [count, children_list].
# - `count`: An integer storing how many strings pass through this node (i.e., have the prefix corresponding to this node).
# - `children_list`: A list of 26 elements, one for each letter 'a'-'z'. Each element can be another node or None.
trie = [0, [None] * 26]

# --- Step 1: Build the Trie ---
# Iterate through each of the N input strings and add them to the Trie.
for _ in range(N):
    # Read a string and remove any trailing whitespace (like the newline character).
    s = input().strip()
    
    # Start traversal from the root of the Trie.
    node = trie
    for char in s:
        # Calculate the index (0-25) corresponding to the character.
        idx = ord(char) - ord('a')
        
        # If a child node for this character doesn't exist, create one.
        if node[1][idx] is None:
            node[1][idx] = [0, [None] * 26]
        
        # Move to the child node.
        node = node[1][idx]
        
        # Increment the counter for this node, as one more string passes through it.
        node[0] += 1

# --- Step 2: Calculate the total sum of LCPs ---
# The core idea is that the sum of LCPs over all pairs can be calculated by summing
# contributions from each node in the Trie.
# The contribution of a single node is the number of pairs of strings that pass through it.
# This is because a pair of strings with an LCP of length k will share a path of k nodes
# from the root, and thus be counted at each of these k nodes.

total_lcp = 0

# We use a deque for an efficient Breadth-First Search (BFS) traversal of the Trie.
queue = collections.deque()

# We start the traversal from the children of the root, as the root represents an
# empty prefix of length 0, which doesn't contribute to the LCP sum.
for child in trie[1]:
    if child is not None:
        queue.append(child)

while queue:
    # Dequeue the next node to process.
    node = queue.popleft()
    
    count = node[0]
    
    # If `count` strings pass through this node, they form C(count, 2) pairs.
    # C(n, 2) = n * (n - 1) / 2.
    # Each of these pairs shares the prefix corresponding to this node.
    # We add this number to our total sum.
    total_lcp += count * (count - 1) // 2
    
    # Enqueue all existing children of the current node to continue the BFS traversal.
    for child_node in node[1]:
        if child_node is not None:
            queue.append(child_node)
            
# Finally, print the calculated total sum.
print(total_lcp)
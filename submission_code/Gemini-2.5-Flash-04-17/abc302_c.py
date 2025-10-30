# YOUR CODE HERE
import sys

def hamming_distance(s1, s2):
    """Calculates the Hamming distance between two strings of equal length."""
    distance = 0
    # Assumes s1 and s2 have the same length M
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            distance += 1
    return distance

# Read input
N, M = map(int, sys.stdin.readline().split())
strings = [sys.stdin.readline().strip() for _ in range(N)]

# This problem can be modeled as finding a Hamiltonian path in a graph
# where nodes are strings and an edge exists between two strings if their
# Hamming distance is exactly 1.
# We can solve this using dynamic programming with bitmask.

# DP state: dp[mask][last_node_index]
# dp[mask][last_node_index] is True if there is a path visiting the nodes
# represented by the bitmask 'mask', ending at the node with index 'last_node_index'.
# The indices of nodes are 0 to N-1, corresponding to the indices in the 'strings' list.
# Initialize with False.
# The size of the DP table is 2^N * N.
dp = [[False] * N for _ in range(1 << N)]

# Base cases: A path consisting of a single node.
# For each node i (0 to N-1), a path starting and ending at i visiting only i exists.
# The mask for visiting only node i is 1 << i.
for i in range(N):
    dp[1 << i][i] = True

# Iterate through masks in increasing order of the number of set bits (path length)
# mask goes from 1 (single node) up to (1 << N) - 1 (all nodes)
# We can iterate simply from 1 to (1 << N) - 1
for mask in range(1, 1 << N):
    # last_node is the index of the last node in the path represented by 'mask'
    for last_node in range(N):
        # Check if a path visiting nodes in 'mask' and ending at 'last_node' exists
        if dp[mask][last_node]:
            # If such a path exists, try to extend it to a 'next_node'
            for next_node in range(N):
                # A 'next_node' must not be in the current mask (not visited yet)
                # Check if the next_node bit is NOT set in the current mask
                if not (mask & (1 << next_node)):
                    # Check if there is a valid edge between the last_node and the next_node
                    # An edge exists if the Hamming distance is exactly 1.
                    if hamming_distance(strings[last_node], strings[next_node]) == 1:
                        # If an edge exists, we can extend the path.
                        # The new mask includes the next_node.
                        new_mask = mask | (1 << next_node)
                        # Mark that a path visiting nodes in new_mask exists, ending at next_node.
                        dp[new_mask][next_node] = True

# After filling the DP table, check if any path visits all nodes.
# A Hamiltonian path visits all N nodes.
# The mask representing all nodes visited is (1 << N) - 1.
found_hamiltonian_path = False
full_mask = (1 << N) - 1
# Check if there is any valid path ending at any node that visits all nodes.
for last_node in range(N):
    if dp[full_mask][last_node]:
        found_hamiltonian_path = True
        break # Found a Hamiltonian path, no need to check other ending nodes

# Print the result
if found_hamiltonian_path:
    print("Yes")
else:
    print("No")
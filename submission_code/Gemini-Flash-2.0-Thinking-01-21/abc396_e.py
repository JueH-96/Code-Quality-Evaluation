import sys
from collections import deque

def solve():
    # Read input N and M
    N, M = map(int, sys.stdin.readline().split())
    
    # Read constraints and store them (adjusting to 0-based indexing)
    constraints = []
    for _ in range(M):
        x, y, z = map(int, sys.stdin.readline().split())
        constraints.append((x - 1, y - 1, z)) 

    # Initialize the result sequence A with zeros
    A = [0] * N

    # Maximum value of Z_i is 10^9. The largest power of 2 less than or equal to 10^9 is 2^29.
    # This means bits up to position 29 might be set. We need to check bit positions
    # from 0 up to 29. To be safe and cover potentially larger values up to just below 2^31,
    # we can check up to bit 30 (as 2^30 > 10^9). Range 0 to 30 inclusive means 31 bits.
    max_bits = 31 

    # Iterate through each bit position k from 0 up to max_bits - 1
    for k in range(max_bits):
        # Build graph for the current bit k
        # The graph has N vertices (0 to N-1). An edge exists between u and v
        # if there's a constraint (u, v, Z_i) or (v, u, Z_i).
        # The weight of the edge (u, v) for bit k is the k-th bit of Z_i.
        adj = [[] for _ in range(N)]
        for u, v, z in constraints:
            weight = (z >> k) & 1 # Get the k-th bit of Z_i
            adj[u].append((v, weight))
            adj[v].append((u, weight))

        # visited_global_for_bit tracks nodes that have been visited and processed 
        # as part of a connected component for the current bit k. This prevents processing
        # the same component multiple times within the loop for bit k.
        visited_global_for_bit = [False] * N 

        # Iterate through all nodes. If a node hasn't been visited yet, 
        # it's the root of a new connected component for this bit k that hasn't been processed.
        for i in range(N):
            if not visited_global_for_bit[i]:
                # Start BFS traversal for a new connected component rooted at node i
                root = i
                # nodes_in_current_component stores all nodes found during the current BFS traversal
                # for this component.
                nodes_in_current_component = []
                # xor_dist_component_local: dictionary storing the XOR sum of edge weights
                # along the path from the component root 'i' to each node 'p' within this component.
                # This value represents the required XOR difference between A_p^(k) and A_root^(k).
                # It's equivalent to A_p^(k) ^ A_root^(k).
                # Using a dictionary implicitly tracks nodes visited within this *specific*
                # component traversal.
                xor_dist_component_local = {} 

                # Initialize BFS queue
                q = deque()

                # Start BFS from the root of the new component
                q.append(root)
                visited_global_for_bit[root] = True # Mark root as visited globally for this bit k
                xor_dist_component_local[root] = 0 # Distance from root to itself is 0
                nodes_in_current_component.append(root)

                # Flag to detect inconsistency within the current component
                consistent = True
                
                # Perform BFS traversal
                while q:
                    u = q.popleft() # Get the next node to process from the queue

                    # Explore neighbors of u
                    for v, weight in adj[u]:
                        # If neighbor v has not been visited in this current component traversal yet
                        if v not in xor_dist_component_local: 
                            # Visit node v
                            visited_global_for_bit[v] = True # Mark v as visited globally for bit k
                            # The required XOR difference from root to v is the XOR difference from root to u
                            # XOR the difference from u to v.
                            xor_dist_component_local[v] = xor_dist_component_local[u] ^ weight
                            nodes_in_current_component.append(v) # Add v to the list of nodes in this component
                            q.append(v) # Add v to the queue for future processing
                        else:
                            # Neighbor v has already been visited in this current component traversal.
                            # This edge (u, v) closes a cycle in the discovered part of the component.
                            # Check for consistency: The XOR difference between A_u^(k) and A_v^(k)
                            # must equal the edge weight 'weight'.
                            # We have A_u^(k) = A_root^(k) ^ xor_dist_component_local[u]
                            # And   A_v^(k) = A_root^(k) ^ xor_dist_component_local[v]
                            # Their XOR is (A_root^(k) ^ xor_dist_component_local[u]) ^ (A_root^(k) ^ xor_dist_component_local[v])
                            # which simplifies to xor_dist_component_local[u] ^ xor_dist_component_local[v].
                            # This value must equal the edge weight 'weight'.
                            if (xor_dist_component_local[u] ^ xor_dist_component_local[v]) != weight:
                                # Inconsistency found: the required XOR relationship along the edge
                                # contradicts the relationship implied by the path distances from the root.
                                print(-1)
                                return # Output -1 and terminate, as no good sequence exists

                # BFS traversal for the current component finished without inconsistency.
                # Now, decide the bit assignments (0 or 1) for each node in this component
                # for bit k to minimize the total sum for this bit.
                # For any node p in the component, A_p^(k) must be either xor_dist_component_local[p]
                # or 1 ^ xor_dist_component_local[p], depending on the choice of A_root^(k) (0 or 1).
                # If A_root^(k) = 0, then A_p^(k) = xor_dist_component_local[p]. The number of 1s
                # for this component for bit k is the count of nodes where xor_dist_component_local is 1 (count1).
                # If A_root^(k) = 1, then A_p^(k) = 1 ^ xor_dist_component_local[p]. The number of 1s
                # for this component for bit k is the count of nodes where xor_dist_component_local is 0 (count0).
                # To minimize the sum for bit k, we choose the assignment (equivalent to choosing A_root^(k))
                # that results in the minimum of count0 and count1 ones.

                count0 = 0 # count of nodes p in the component where xor_dist_component_local[p] == 0 (relative to root)
                count1 = 0 # count of nodes p in the component where xor_dist_component_local[p] == 1 (relative to root)

                for node in nodes_in_current_component:
                    if xor_dist_component_local[node] == 0:
                        count0 += 1
                    else:
                        count1 += 1

                # Determine the actual bit value (0 or 1) that should be assigned to nodes
                # based on their relative distance from the root (0 or 1) and the counts.
                # If count1 <= count0, choosing the assignment where nodes with xor_dist 0 get bit 0
                # (and nodes with xor_dist 1 get bit 1) results in fewer or equal 1s. This corresponds
                # to setting A_root^(k) = 0.
                # If count0 < count1, choosing the assignment where nodes with xor_dist 0 get bit 1
                # (and nodes with xor_dist 1 get bit 0) results in fewer 1s. This corresponds
                # to setting A_root^(k) = 1.
                
                assign_bit_for_dist0 = 0 # The bit value assigned to a node if its xor_dist from root is 0
                assign_bit_for_dist1 = 1 # The bit value assigned to a node if its xor_dist from root is 1

                if count0 < count1: 
                    # Flipped assignment is better (A_p^(k) = 1 ^ xor_dist[p])
                    assign_bit_for_dist0 = 1
                    assign_bit_for_dist1 = 0

                # Apply the decided bit assignments for this component for bit k
                power_of_2k = (1 << k)
                for node in nodes_in_current_component:
                    if xor_dist_component_local[node] == 0:
                        bit_val = assign_bit_for_dist0
                    else:
                        bit_val = assign_bit_for_dist1
                    
                    # Add the contribution of this bit value (0 or 1) multiplied by 2^k to the total value of A[node]
                    A[node] += bit_val * power_of_2k

    # If the loop finishes for all bit positions without encountering any inconsistency,
    # we have successfully constructed a good sequence A that minimizes the sum.
    # Print the elements of A separated by spaces.
    print(*A)

# Execute the solve function
solve()
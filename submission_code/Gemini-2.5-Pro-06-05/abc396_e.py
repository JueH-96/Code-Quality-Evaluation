# YOUR CODE HERE
import sys

def solve():
    """
    Solves the Good Sequence problem by modeling it as a graph problem.
    It finds a non-negative integer sequence A that satisfies given XOR constraints
    and minimizes the sum of its elements.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        
        N, M = map(int, input().split())
        
        adj = [[] for _ in range(N + 1)]
        for _ in range(M):
            X, Y, Z = map(int, input().split())
            adj[X].append((Y, Z))
            adj[Y].append((X, Z))

    except (IOError, ValueError):
        # This handles cases like empty input. For this problem, the input format
        # is guaranteed, but this makes the code more robust.
        return

    visited = [False] * (N + 1)
    ans_A = [0] * (N + 1)
    # potential[i] stores the value of A[i] relative to the root of its component.
    # Specifically, potential[i] = A[i] ^ A[root].
    potential = [0] * (N + 1)
    
    # Iterate through all nodes to handle disconnected components.
    for i in range(1, N + 1):
        if not visited[i]:
            # Found a new connected component starting at node i.
            
            # We use a list as a queue for BFS, with a 'head' pointer for efficiency.
            # This list will also serve as our list of component nodes.
            component_nodes = [i]
            visited[i] = True
            potential[i] = 0  # The root's potential relative to itself is 0.
            head = 0
            possible = True

            # BFS to traverse the component, calculate potentials, and check for consistency.
            while head < len(component_nodes):
                u = component_nodes[head]
                head += 1
                
                for v, z in adj[u]:
                    if not visited[v]:
                        # First time visiting node v.
                        visited[v] = True
                        # The potential is determined by the path from the root.
                        # A[v] = A[u] ^ z
                        # A[v]^A[root] = (A[u]^A[root]) ^ z
                        potential[v] = potential[u] ^ z
                        component_nodes.append(v)
                    elif (potential[u] ^ potential[v]) != z:
                        # Reached an already visited node. Check for consistency.
                        # The relationship A[u]^A[v] must equal z.
                        # This is equivalent to (A[u]^A[root])^(A[v]^A[root]) == z
                        possible = False
                        break
                if not possible:
                    break
            
            if not possible:
                print(-1)
                return

            # If the component is consistent, find the optimal base value 'base_C'
            # to minimize the sum of elements in this component.
            # A[node] = base_C ^ potential[node] for all nodes in the component.
            base_C = 0
            
            # We check bit by bit from Most Significant to Least Significant.
            # max(Z_i) is 10^9 < 2^30, so bit 29 is the highest possible.
            for k in range(29, -1, -1):
                c1 = 0  # Count of nodes where k-th bit of potential is 1
                for node in component_nodes:
                    if (potential[node] >> k) & 1:
                        c1 += 1
                
                c0 = len(component_nodes) - c1
                
                # To minimize the sum contribution from this bit, we choose the
                # bit for base_C that flips the bit in the potentials of the
                # minority group.
                # Contribution if base_C[k]=0: c1 * (2^k)
                # Contribution if base_C[k]=1: c0 * (2^k)
                if c1 > c0:
                    base_C |= (1 << k)
            
            # Calculate the final values for the nodes in this component.
            for node in component_nodes:
                ans_A[node] = base_C ^ potential[node]

    # Print the final result.
    print(*ans_A[1:])

if __name__ == "__main__":
    solve()
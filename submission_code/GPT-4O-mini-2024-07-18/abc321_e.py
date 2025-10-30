def count_vertices(N, X, K):
    if K == 0:
        return 1 if X <= N else 0
    
    # Calculate the number of vertices at distance K from X
    count = 0
    # Level of the current node
    level = 0
    # Current node to explore
    current = X
    
    # Traverse upwards to find the path to the root
    path = []
    while current > 0:
        path.append(current)
        current //= 2
    
    # The number of nodes at distance K from X can be calculated
    # based on the level of X and the tree structure
    for i in range(len(path)):
        if level + K < len(path):
            # Count nodes at distance K downwards from the current node
            count += (1 << (K))  # 2^K nodes at this level
        elif level + K == len(path):
            # Count only the current node
            count += 1
        else:
            # We are going above the root, no nodes to count
            break
        level += 1
    
    # Now we need to subtract nodes that are out of bounds (greater than N)
    # We will check the nodes at distance K from X
    if K > 0:
        # Check the nodes at distance K from the path
        for i in range(len(path)):
            if level + K < len(path):
                # Nodes below the current path node
                count += (1 << (K))  # 2^K nodes at this level
            elif level + K == len(path):
                count += 1
            else:
                break
            level += 1
    
    # Return the count of valid nodes
    return count

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    T = int(data[0])
    results = []
    
    for i in range(1, T + 1):
        N, X, K = map(int, data[i].split())
        result = count_vertices(N, X, K)
        results.append(result)
    
    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()
import sys
import math

def solve():
    input = sys.stdin.read
    data = input().split()
    
    T = int(data[0])
    index = 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        X = int(data[index + 1])
        K = int(data[index + 2])
        index += 3
        
        # Calculate depth of X
        depth_X = X.bit_length() - 1
        
        if K <= depth_X:
            # Move K steps upwards
            ancestor = X
            for _ in range(K):
                ancestor //= 2
            # Calculate the number of nodes in the subtree rooted at ancestor
            max_depth = depth_X - K
            max_depth = min(max_depth, N.bit_length() - 1)
            max_nodes = (1 << (max_depth + 1)) - 1
            if ancestor * 2 <= N:
                max_nodes -= (ancestor * 2 - 1)
            if ancestor * 2 + 1 <= N:
                max_nodes -= (ancestor * 2 + 1 - 1)
            results.append(max_nodes)
        else:
            # Move to the root and then K - depth_X steps downwards
            remaining_depth = K - depth_X
            if remaining_depth < 0:
                results.append(0)
            else:
                # Calculate the number of nodes at this depth
                if remaining_depth >= N.bit_length():
                    results.append(0)
                else:
                    max_nodes = 1 << remaining_depth
                    if (1 << remaining_depth) > N:
                        max_nodes = max(0, N - (1 << remaining_depth) + 1)
                    results.append(max_nodes)
    
    sys.stdout.write("
".join(map(str, results)) + "
")
# YOUR CODE HERE
import sys
import math

input = sys.stdin.read
data = input().split()

T = int(data[0])
results = []
index = 1

for _ in range(T):
    N = int(data[index])
    X = int(data[index + 1])
    K = int(data[index + 2])
    index += 3
    
    # Calculate the depth of X
    depth_X = int(math.log2(X))
    
    # Calculate the target depth
    target_depth = depth_X + K
    
    if target_depth < 0:
        results.append(0)
        continue
    
    # Calculate the range of nodes at the target depth
    start_node = 2 ** target_depth
    end_node = 2 ** (target_depth + 1) - 1
    
    if start_node > N:
        results.append(0)
    else:
        results.append(min(end_node, N) - start_node + 1)

print("
".join(map(str, results)))
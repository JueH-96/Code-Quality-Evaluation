# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def main():
    N = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    # Adjust to 0-based index for easier handling
    a = [x-1 for x in a]
    
    visited = [False] * N
    reachable = [0] * N
    
    for i in range(N):
        if not visited[i]:
            cycle = []
            stack = []
            current = i
            while True:
                if visited[current]:
                    if current in stack:
                        # Found a cycle
                        cycle_start = stack.index(current)
                        cycle = stack[cycle_start:]
                        # All nodes in the cycle can reach each other
                        cycle_size = len(cycle)
                        for node in cycle:
                            reachable[node] = cycle_size
                        # Now, handle the nodes before the cycle
                        for idx in range(cycle_start):
                            reachable[stack[idx]] = cycle_size + (cycle_start - idx)
                    else:
                        # No cycle, just a path to a visited node
                        # All nodes in the stack can reach the reachable[current] nodes
                        for idx in range(len(stack)):
                            reachable[stack[idx]] = reachable[current] + (len(stack) - idx)
                    break
                visited[current] = True
                stack.append(current)
                current = a[current]
    
    total = sum(reachable)
    print(total)

if __name__ == "__main__":
    main()
# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    P = list(map(int, data[2:N+1]))
    XY = list(map(int, data[N+2:]))
    
    # Build the family tree
    tree = defaultdict(list)
    for i, p in enumerate(P, 2):
        tree[p].append(i)
    
    # Function to get all descendants of a person up to y generations
    def get_descendants(person, y):
        descendants = set()
        queue = deque([(person, 0)])
        while queue:
            current, generation = queue.popleft()
            if generation > y:
                break
            for child in tree[current]:
                descendants.add(child)
                queue.append((child, generation + 1))
        return descendants
    
    # Track covered people
    covered = set()
    for i in range(M):
        x = XY[2*i]
        y = XY[2*i + 1]
        descendants = get_descendants(x, y)
        covered.add(x)
        covered.update(descendants)
    
    print(len(covered))

if __name__ == "__main__":
    solve()
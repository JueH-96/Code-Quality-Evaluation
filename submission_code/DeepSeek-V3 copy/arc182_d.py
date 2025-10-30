import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Check if the sequences are already the same
    if A == B:
        print(0)
        return
    
    # Precompute the possible values for each position
    # For each position i, the value must be different from A[i-1] and A[i+1]
    # But since we are transforming A to B, we need to ensure that B[i] is different from B[i-1] and B[i+1]
    # However, the problem states that B is already a good sequence, so we don't need to check that
    
    # We need to find the minimum number of operations to transform A to B, ensuring that after each operation, A remains a good sequence
    
    # We can model this as a graph where each node represents a possible state of A
    # However, the state space is too large (M^N), so we need a smarter approach
    
    # Instead, we can consider each position independently, but ensuring that the changes do not violate the good sequence condition
    
    # For each position i, we need to find the minimum number of operations to change A[i] to B[i], while ensuring that A[i] != A[i-1] and A[i] != A[i+1] after each operation
    
    # We can precompute the possible values for each position, considering the constraints from the neighboring positions
    
    # Let's create a list of possible values for each position
    possible = []
    for i in range(N):
        if i == 0:
            # Only need to be different from A[1]
            possible.append([x for x in range(M) if x != A[1]])
        elif i == N-1:
            # Only need to be different from A[N-2]
            possible.append([x for x in range(M) if x != A[N-2]])
        else:
            # Need to be different from A[i-1] and A[i+1]
            possible.append([x for x in range(M) if x != A[i-1] and x != A[i+1]])
    
    # Now, for each position, we need to find the minimum number of operations to change A[i] to B[i], considering the possible values
    
    # We can use BFS for each position to find the minimum number of operations
    
    total_ops = 0
    for i in range(N):
        if A[i] == B[i]:
            continue
        # BFS to find the minimum number of operations to change A[i] to B[i], considering the possible values
        visited = set()
        queue = deque()
        queue.append((A[i], 0))
        visited.add(A[i])
        found = False
        while queue:
            current, ops = queue.popleft()
            if current == B[i]:
                total_ops += ops
                found = True
                break
            # Try +1 and -1 operations
            next_val = (current + 1) % M
            if next_val in possible[i] and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, ops + 1))
            next_val = (current - 1) % M
            if next_val in possible[i] and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, ops + 1))
        if not found:
            print(-1)
            return
    print(total_ops)

if __name__ == "__main__":
    main()
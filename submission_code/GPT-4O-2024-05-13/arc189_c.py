# YOUR CODE HERE
import sys
from collections import deque

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1]) - 1
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:2*N+2]))
    P = list(map(lambda x: int(x) - 1, data[2*N+2:3*N+2]))
    Q = list(map(lambda x: int(x) - 1, data[3*N+2:4*N+2]))
    
    if all(a == 0 and b == 0 for a, b in zip(A, B)):
        print(0)
        return
    
    visited = set()
    queue = deque([(tuple(A), tuple(B), 0)])
    
    while queue:
        current_A, current_B, steps = queue.popleft()
        
        if all(i == X or (current_A[i] == 0 and current_B[i] == 0) for i in range(N)):
            print(steps)
            return
        
        for i in range(N):
            if current_A[i] > 0 or current_B[i] > 0:
                new_A = list(current_A)
                new_B = list(current_B)
                
                new_A[P[i]] += new_A[i]
                new_B[Q[i]] += new_B[i]
                
                new_A[i] = 0
                new_B[i] = 0
                
                new_state = (tuple(new_A), tuple(new_B))
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state[0], new_state[1], steps + 1))
    
    print(-1)

if __name__ == "__main__":
    solve()
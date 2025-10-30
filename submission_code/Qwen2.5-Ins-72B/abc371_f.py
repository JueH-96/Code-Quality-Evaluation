# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = list(map(int, data[1:N+1]))
    Q = int(data[N+1])
    tasks = [list(map(int, data[N+2+i*2:N+4+i*2])) for i in range(Q)]
    
    total_moves = 0
    positions = X[:]
    
    for T, G in tasks:
        T -= 1
        current_pos = positions[T]
        moves = abs(G - current_pos)
        total_moves += moves
        
        # Update the positions of all persons
        for i in range(N):
            if i < T:
                positions[i] += 1 if G > current_pos else -1
            elif i > T:
                positions[i] += 1 if G > current_pos else -1
        positions[T] = G
    
    print(total_moves)

if __name__ == "__main__":
    solve()
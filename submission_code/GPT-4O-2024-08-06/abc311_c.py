# YOUR CODE HERE
def find_cycle(N, A):
    visited = [False] * (N + 1)
    cycle = []

    for start in range(1, N + 1):
        if not visited[start]:
            current = start
            path = []
            while not visited[current]:
                visited[current] = True
                path.append(current)
                current = A[current - 1]
            
            # Once we revisit a node, we have found a cycle
            # Find the start of the cycle in the path
            cycle_start = path.index(current)
            cycle = path[cycle_start:]
            break

    return cycle

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    cycle = find_cycle(N, A)
    
    print(len(cycle))
    print(" ".join(map(str, cycle)))
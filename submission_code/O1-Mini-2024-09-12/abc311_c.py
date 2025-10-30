# YOUR CODE HERE
import sys
import sys
import sys

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    visited = [False] * (N + 1)
    in_stack = [False] * (N + 1)
    path = []
    for i in range(1, N+1):
        if not visited[i]:
            current = i
            while True:
                visited[current] = True
                in_stack[current] = True
                path.append(current)
                next_node = A[current-1]
                if in_stack[next_node]:
                    # Found a cycle
                    idx = path.index(next_node)
                    cycle = path[idx:]
                    print(len(cycle))
                    print(' '.join(map(str, cycle)))
                    return
                elif visited[next_node]:
                    break
                else:
                    current = next_node
            # Remove the current path from stack
            while path:
                node = path.pop()
                in_stack[node] = False

if __name__ == "__main__":
    main()
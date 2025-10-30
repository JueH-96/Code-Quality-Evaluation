# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+M]))
    B = list(map(int, input[2+M:2+2*M]))

    # Create a set to track conflicts
    conflicts = set()
    for i in range(M):
        if A[i] < B[i]:
            conflicts.add((A[i], B[i]))
        else:
            conflicts.add((B[i], A[i]))

    # Check for cycles in the conflicts
    visited = [False] * (N + 1)
    stack = [False] * (N + 1)

    def is_cyclic(v):
        stack[v] = True
        visited[v] = True

        for u in range(1, N + 1):
            if (v, u) in conflicts:
                if not visited[u]:
                    if is_cyclic(u):
                        return True
                elif stack[u]:
                    return True

        stack[v] = False
        return False

    for i in range(1, N + 1):
        if not visited[i]:
            if is_cyclic(i):
                print("No")
                return

    print("Yes")

if __name__ == "__main__":
    main()
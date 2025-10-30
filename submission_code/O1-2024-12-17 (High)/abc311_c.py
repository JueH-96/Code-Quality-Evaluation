def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = int(data[i])

    visited = [False] * (N + 1)

    # We only need to find one cycle, then print it and return.
    for start in range(1, N + 1):
        if not visited[start]:
            stack = []
            index_in_stack = {}
            current = start
            while True:
                if current in index_in_stack:
                    # A cycle is found. Extract it from where current was first seen.
                    cycle_start_pos = index_in_stack[current]
                    cycle = stack[cycle_start_pos:]
                    print(len(cycle))
                    print(" ".join(map(str, cycle)))
                    return

                # Mark current's position in the visiting stack
                index_in_stack[current] = len(stack)
                stack.append(current)
                visited[current] = True

                # Move to the next vertex
                nxt = A[current]

                # If nxt is already visited but not in the current chain => no cycle here
                if visited[nxt] and nxt not in index_in_stack:
                    break

                current = nxt

# Do not forget to call main.
if __name__ == "__main__":
    main()
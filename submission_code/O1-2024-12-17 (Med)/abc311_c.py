def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(lambda x: int(x) - 1, input_data[1:]))

    visited = [0] * N  # 0: unvisited, 1: in current path, 2: fully processed

    for start in range(N):
        if visited[start] == 0:
            path = []
            index_in_path = {}
            current = start

            # Follow the chain until we revisit a node
            while visited[current] == 0:
                visited[current] = 1
                index_in_path[current] = len(path)
                path.append(current)
                current = A[current]

            # If we found a node that is in the current path -> cycle detected
            if visited[current] == 1:
                cycle_start = index_in_path[current]
                cycle = path[cycle_start:]
                print(len(cycle))
                print(' '.join(str(x + 1) for x in cycle))
                return

            # Mark all nodes in path as fully processed
            for node in path:
                visited[node] = 2

def run():
    main()

# Call main() directly as required.
if __name__ == "__main__":
    main()
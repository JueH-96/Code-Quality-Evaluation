def find_cycle(start, graph):
    cycle = []
    while True:
        cycle.append(start)
        next_node = graph[start]
        if next_node in cycle:
            break
        start = next_node
    return cycle

def apply_permutation(A, cycles):
    for cycle in cycles:
        cycle_values = [A[node - 1] for node in cycle]
        cycle_values.sort()
        for i, node in enumerate(cycle):
            A[node - 1] = cycle_values[i]

def main():
    N = int(input().strip())
    P = list(map(int, input().strip().split()))
    A = list(map(int, input().strip().split()))

    # Create a graph representation of the permutation
    graph = {i + 1: P[i] for i in range(N)}

    # Find all cycles in the permutation
    visited = set()
    cycles = []
    for i in range(1, N + 1):
        if i not in visited:
            cycle = find_cycle(i, graph)
            cycles.append(cycle)
            visited.update(cycle)

    # Apply the permutation to get the lexicographically smallest A
    apply_permutation(A, cycles)

    # Print the result
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()
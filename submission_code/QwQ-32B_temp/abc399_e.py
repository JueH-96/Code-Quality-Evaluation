import sys

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()

    # Check if the mapping is consistent
    s_to_t = {}
    possible = True
    for sc, tc in zip(s, t):
        if sc in s_to_t:
            if s_to_t[sc] != tc:
                possible = False
                break
        else:
            s_to_t[sc] = tc
    if not possible:
        print(-1)
        return

    # Collect all characters present in S
    C = set(s)
    # Build the mapping function f
    f = {c: s_to_t[c] for c in C}

    # Calculate edges_count: number of characters with f[c] != c
    edges_count = sum(1 for c in C if f[c] != c)

    # Function to count cycles in the functional graph
    def count_cycles(f, C):
        visited = set()
        cycles = 0
        for node in C:
            if node not in visited:
                path = []
                current = node
                while current in C and current not in visited:
                    visited.add(current)
                    path.append(current)
                    current = f[current]
                # Check if current is in the path (indicating a cycle)
                if current in path:
                    cycles += 1
        return cycles

    cycles_count = count_cycles(f, C)
    print(edges_count + cycles_count)

if __name__ == "__main__":
    main()
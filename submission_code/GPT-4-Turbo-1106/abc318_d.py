from itertools import combinations

def max_weight(N, weights):
    # Create a list of all possible pairs of vertices
    vertices = list(range(N))
    pairs = list(combinations(vertices, 2))

    # Initialize the maximum weight to 0
    max_weight = 0

    # Use bitmasking to generate all possible combinations of edges
    for mask in range(1 << len(pairs)):
        total_weight = 0
        chosen_vertices = set()

        # Check each pair to see if it is included in the current combination
        for i, (u, v) in enumerate(pairs):
            if mask & (1 << i):
                # If either vertex is already chosen, skip this pair
                if u in chosen_vertices or v in chosen_vertices:
                    break
                # Add the weight of the current edge and mark the vertices as chosen
                total_weight += weights[u][v]
                chosen_vertices.add(u)
                chosen_vertices.add(v)
        else:
            # Update the maximum weight if the current total weight is greater
            max_weight = max(max_weight, total_weight)

    return max_weight

def main():
    N = int(input().strip())
    weights = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        row = list(map(int, input().split()))
        for j, weight in enumerate(row, start=i+1):
            weights[i][j] = weight
    print(max_weight(N, weights))

if __name__ == "__main__":
    main()
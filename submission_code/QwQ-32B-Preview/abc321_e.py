import math

def count_vertices_at_distance(N, X, K):
    if N < 1 or X < 1 or X > N or K < 0 or K > N - 1:
        return 0

    # Function to calculate depth of a vertex
    def depth(I):
        return 0 if I == 1 else depth(I // 2) + 1

    dx = depth(X)
    max_depth = depth(N)
    
    count = 0
    for U in range(min(K, dx) + 1):
        D = dx + K - 2 * U
        if D < 0 or D > max_depth:
            continue
        # Calculate the ancestor by moving U steps up from X
        ancestor = X
        for _ in range(U):
            ancestor = ancestor // 2
        # Number of vertices at depth D in the subtree rooted at ancestor
        # Adjust for the actual number of nodes in the tree
        nodes_at_D = min(1 << D, N - (1 << D) + 1)
        # If ancestor is not the root, subtract overlapping nodes
        if ancestor > 1:
            # Calculate the depth of ancestor
            d_ancestor = dx - U
            # Nodes in subtrees of ancestor's siblings
            if D < d_ancestor:
                nodes_at_D = 0
            else:
                nodes_at_D -= 1 << (D - d_ancestor - 1)
        count += nodes_at_D
    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    index = 1
    for _ in range(T):
        N = int(data[index])
        X = int(data[index + 1])
        K = int(data[index + 2])
        index += 3
        print(count_vertices_at_distance(N, X, K))

if __name__ == "__main__":
    main()
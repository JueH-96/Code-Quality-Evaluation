import sys

def solve(N, X, K):
    """
    Calculate the number of vertices whose distance from vertex X is K.

    Args:
    N (int): The number of vertices in the tree.
    X (int): The vertex from which to calculate distances.
    K (int): The distance from vertex X.

    Returns:
    int: The number of vertices whose distance from vertex X is K.
    """
    if K == 0:
        return 1

    # Calculate the number of vertices at each level
    level_counts = [1]
    for i in range(1, 64):  # 64 is sufficient for N up to 10^18
        level_counts.append(min(N + 1, level_counts[-1] * 2))

    # Find the level of vertex X
    X_level = 0
    while X > level_counts[X_level]:
        X_level += 1

    # Calculate the number of vertices at the same level as X
    same_level_count = min(N + 1, level_counts[X_level]) - level_counts[X_level - 1]

    # Calculate the number of vertices at the level K above X
    if X_level + K >= len(level_counts):
        return 0
    above_count = min(N + 1, level_counts[X_level + K]) - level_counts[X_level + K - 1]

    # Calculate the number of vertices at the level K below X
    if X_level - K < 0:
        return above_count
    below_count = same_level_count

    # Calculate the number of vertices whose distance from X is K
    if K % 2 == 0:
        return above_count
    else:
        return below_count

def main():
    T = int(sys.stdin.readline().strip())
    for _ in range(T):
        N, X, K = map(int, sys.stdin.readline().strip().split())
        result = solve(N, X, K)
        sys.stdout.write(str(result) + '
')

if __name__ == '__main__':
    main()
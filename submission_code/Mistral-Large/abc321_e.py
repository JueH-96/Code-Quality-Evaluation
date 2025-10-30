import sys
import math

def count_vertices_at_distance(N, X, K):
    if K == 0:
        return 1

    current_level = [X]
    next_level = []
    distance = 0

    while distance < K and current_level:
        for node in current_level:
            left_child = 2 * node
            right_child = 2 * node + 1
            if left_child <= N:
                next_level.append(left_child)
            if right_child <= N:
                next_level.append(right_child)
        current_level = next_level
        next_level = []
        distance += 1

    return len(current_level)

def main():
    input = sys.stdin.read
    data = input().split()

    T = int(data[0])
    index = 1
    results = []

    for _ in range(T):
        N = int(data[index])
        X = int(data[index + 1])
        K = int(data[index + 2])
        index += 3
        results.append(count_vertices_at_distance(N, X, K))

    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == "__main__":
    main()
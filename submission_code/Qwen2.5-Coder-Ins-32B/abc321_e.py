# YOUR CODE HERE
import sys

def count_vertices(N, X, K):
    if K == 0:
        return 1
    
    def count_children(node, depth):
        if depth == K:
            return 1
        if node * (2 ** (K - depth)) > N:
            return 0
        left_child = node * 2
        right_child = node * 2 + 1
        left_count = count_children(left_child, depth + 1)
        right_count = count_children(right_child, depth + 1)
        return left_count + right_count

    return count_children(X, 0)

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    T = int(input[index])
    index += 1
    results = []
    for _ in range(T):
        N = int(input[index])
        X = int(input[index + 1])
        K = int(input[index + 2])
        index += 3
        results.append(count_vertices(N, X, K))
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
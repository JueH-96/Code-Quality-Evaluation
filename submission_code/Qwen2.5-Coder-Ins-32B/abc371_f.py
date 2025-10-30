# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    index += 1
    X = list(map(int, input[index:index + N]))
    index += N
    Q = int(input[index])
    index += 1
    tasks = []
    for _ in range(Q):
        T_i = int(input[index]) - 1
        G_i = int(input[index + 1])
        tasks.append((T_i, G_i))
        index += 2

    X.sort()
    total_moves = 0
    for T_i, G_i in tasks:
        pos = X[T_i]
        total_moves += abs(G_i - pos)
        # Remove the person from the list and insert at the new position
        X.pop(T_i)
        bisect.insort(X, G_i)

    print(total_moves)

if __name__ == "__main__":
    main()
import sys

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    X = [int(data[2 + i]) for i in range(M)]
    A = [int(data[2 + M + i]) for i in range(M)]

    # Check if the total number of stones is exactly N
    total_stones = sum(A)
    if total_stones != N:
        print(-1)
        return

    # Calculate the minimum number of operations required
    operations = 0
    current_stones = 0

    for i in range(M):
        if X[i] != current_stones + 1:
            # If there is a gap, we need to move stones to fill the gap
            gap = X[i] - (current_stones + 1)
            operations += gap
            current_stones += gap

        # Move the necessary stones to the next cells
        if A[i] > 1:
            operations += (A[i] - 1)
            current_stones += (A[i] - 1)

        current_stones += 1

    print(operations)

if __name__ == "__main__":
    solve()
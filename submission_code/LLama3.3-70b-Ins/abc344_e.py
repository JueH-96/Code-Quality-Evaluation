from collections import deque

def solve():
    # Read the initial sequence
    N = int(input())
    A = deque(map(int, input().split()))

    # Process queries
    Q = int(input())
    for _ in range(Q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            # Insert y immediately after x
            x, y = query[1], query[2]
            idx = list(A).index(x)
            A.insert(idx + 1, y)
        elif query[0] == 2:
            # Remove x
            x = query[1]
            A.remove(x)

    # Print the final sequence
    print(*A)

if __name__ == "__main__":
    solve()
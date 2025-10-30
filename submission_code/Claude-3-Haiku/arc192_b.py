from collections import defaultdict

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Initialize the set of visited indices
    visited = set()

    # Determine the winner
    turn = 0
    while True:
        # Find the next available index
        for i in range(N):
            if A[i] > 0 and i not in visited:
                break
        else:
            # All indices have been visited, so the game ends
            return "Fennec" if turn % 2 == 0 else "Snuke"

        # Perform the operation
        A[i] -= 1
        visited.add(i)

        # Switch turns
        turn += 1

print(solve())
# YOUR CODE HERE

def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))

    steps = 0
    while True:
        max_empty = max(A) if A else 0
        if max_empty <= K:
            A.remove(max_empty)
            K -= max_empty
            steps += 1
        else:
            break

    print(steps)

solve()
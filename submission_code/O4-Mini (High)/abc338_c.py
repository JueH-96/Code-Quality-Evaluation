import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    Q = list(map(int, input().split()))
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Indices where A[i] > 0 (to bound x) and B[i] > 0 (to bound y)
    posA = [i for i in range(N) if A[i] > 0]
    posB = [i for i in range(N) if B[i] > 0]

    # Maximum possible x by ingredient constraints of A
    x_max = min(Q[i] // A[i] for i in posA)

    ans = 0
    # Try all x from 0 to x_max
    for x in range(x_max + 1):
        # For each ingredient with B[i]>0, compute how many servings of B we can make
        # after using x servings of A
        y_max = min((Q[i] - A[i] * x) // B[i] for i in posB)
        # Update the answer with x + y_max
        total = x + y_max
        if total > ans:
            ans = total

    print(ans)

# Call main to execute
main()
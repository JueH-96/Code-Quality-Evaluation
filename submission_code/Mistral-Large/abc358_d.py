import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    A = list(map(int, data[index:index + N]))
    index += N
    B = list(map(int, data[index:index + M]))

    # Sort A and B
    A.sort()
    B.sort()

    # Check if it's possible to satisfy the condition
    for i in range(M):
        if B[i] > A[i]:
            print(-1)
            return

    # Calculate the minimum total amount of money
    total_cost = sum(A[-M:])
    print(total_cost)

if __name__ == "__main__":
    solve()
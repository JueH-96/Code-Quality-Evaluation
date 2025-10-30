import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    A = []
    B = []
    for i in range(1, N + 1):
        A.append(int(data[2 * i - 1]))
        B.append(int(data[2 * i]))

    # Calculate the difference between head height and shoulder height
    diff = [B[i] - A[i] for i in range(N)]

    # Sort the giants based on the difference in descending order
    sorted_diff = sorted(diff, reverse=True)

    # Calculate the maximum possible height
    max_height = sum(A) + sum(sorted_diff[:N])

    print(max_height)

if __name__ == "__main__":
    solve()
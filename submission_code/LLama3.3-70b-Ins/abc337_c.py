import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Find the person at the front of the line
    front = -1
    for i in range(N):
        if A[i] == -1:
            front = i + 1
            break

    # Build the line
    line = [front]
    for _ in range(N - 1):
        for i in range(N):
            if A[i - 1] == line[-1]:
                line.append(i + 1)

    # Print the line
    print(' '.join(map(str, line)))

if __name__ == '__main__':
    solve()
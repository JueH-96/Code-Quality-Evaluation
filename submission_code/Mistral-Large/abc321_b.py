import sys

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))

    A.sort()

    # Calculate the sum of scores from A_2 to A_{N-2}
    current_sum = sum(A[1:N-1])

    # Check if the target score X can be achieved
    if current_sum + 100 < X:
        print(-1)
        return

    # Calculate the minimum score needed in the last round
    needed_score = X - current_sum

    # The needed score should be at least the second lowest score in A
    min_required_score = max(A[1], needed_score)

    print(min_required_score)

solve()
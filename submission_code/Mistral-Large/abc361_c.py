import sys

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))

    # Sort the sequence A
    A.sort()

    # The minimum possible value of the maximum value of B minus the minimum value of B
    # can be achieved by removing the K smallest and K largest elements.
    # The result will be the difference between the (K+1)-th smallest element and
    # the (N-K)-th smallest element.
    result = A[N-K-1] - A[K]

    print(result)

if __name__ == "__main__":
    solve()
# YOUR CODE HERE
def solve():
    # Read N, the length of the sequences
    N = int(input())

    # Read sequence A
    # input().split() reads the line of space-separated numbers as a list of strings.
    # map(int, ...) converts each string in the list to an integer.
    # list(...) converts the map object into a list.
    A = list(map(int, input().split()))

    # Read sequence B
    B = list(map(int, input().split()))

    # Find the maximum element in sequence A.
    # Since the problem constraint states 1 <= N, the lists A and B will not be empty,
    # so max() can be safely called.
    max_A = max(A)

    # Find the maximum element in sequence B.
    max_B = max(B)

    # The maximum possible value of A_i + B_j is the sum of the maximums of A and B.
    result = max_A + max_B

    # Print the result.
    print(result)

if __name__ == '__main__':
    solve()
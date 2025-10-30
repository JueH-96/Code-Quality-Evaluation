def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # The sum of all final scores equals 0.
    # Therefore, the final score of person N is the negative sum of A_1, A_2, ..., A_{N-1}.
    score_of_N = -sum(A)
    print(score_of_N)

# Do not remove the call to main()
main()
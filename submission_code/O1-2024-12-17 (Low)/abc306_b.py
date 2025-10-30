def main():
    import sys

    # Read the 64 binary digits from stdin
    A = list(map(int, sys.stdin.read().split()))

    # Calculate the resulting integer value
    answer = 0
    for i, bit in enumerate(A):
        answer += bit * (2 ** i)

    # Print the result
    print(answer)

# Do not remove next line
main()
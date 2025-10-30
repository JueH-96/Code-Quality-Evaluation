def main():
    import sys

    A, B = map(int, sys.stdin.readline().strip().split())

    # Suspects are 1, 2, 3
    # A is not the culprit, B is not the culprit.
    # We find possible culprits from {1,2,3} excluding A and B.

    possible_culprits = {1, 2, 3} - {A, B}

    if len(possible_culprits) == 1:
        print(possible_culprits.pop())
    else:
        print(-1)

# Do not forget to call main()
main()
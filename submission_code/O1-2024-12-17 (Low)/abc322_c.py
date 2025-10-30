def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:]))

    # We'll iterate i from 1 to N, using a pointer (idx) to point to the 
    # nearest upcoming (or current) fireworks day in the list A.
    idx = 0
    for i in range(1, N + 1):
        # If the current fireworks day is strictly before i, move to the next fireworks day
        while A[idx] < i:
            idx += 1
        # The answer for day i is how many days until the fireworks day aligned by A[idx]
        print(A[idx] - i)

# Do not forget to call main().
main()
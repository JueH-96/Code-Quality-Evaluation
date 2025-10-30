def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:]))

    p = 0  # pointer to the current firework day in A
    for i in range(1, N+1):
        # Move the pointer if the current firework day is strictly before i
        while p < M - 1 and A[p] < i:
            p += 1
        # Print the difference between the next (or current) firework day and i
        print(A[p] - i)

# Do not remove the following line
main()
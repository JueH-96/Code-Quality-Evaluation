import sys
import threading

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    count = 0
    i = 0
    # Two-pointer: for each j as the bottom mochi, move i to count tops
    for j in range(n):
        # Increase i while A[i] * 2 <= A[j]
        # i only moves forward, total moves O(n)
        while i < n and A[i] * 2 <= A[j]:
            i += 1
        # i is now the number of valid top choices for j
        count += i

    # Since A[j] cannot be used as its own top (A[j] > A[j]/2 for positive A[j]),
    # we don't need to subtract any self-pair.
    # Output the total count
    print(count)

if __name__ == "__main__":
    main()
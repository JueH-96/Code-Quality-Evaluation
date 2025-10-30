import sys
import threading

def main():
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # Dictionary to store the most recent index of each value
    last_occurrence = {}
    B = [-1] * n

    for i in range(n):
        x = A[i]
        if x in last_occurrence:
            B[i] = last_occurrence[x]
        else:
            B[i] = -1
        # Update the most recent occurrence of x to the current position (1-based)
        last_occurrence[x] = i + 1

    # Output the results
    print(" ".join(map(str, B)))

if __name__ == "__main__":
    main()
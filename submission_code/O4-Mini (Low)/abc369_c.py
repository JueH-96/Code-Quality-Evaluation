import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # For a single element, only one subarray exists
    if n == 1:
        print(1)
        return

    # Build the difference array D of length n-1
    D = [A[i+1] - A[i] for i in range(n-1)]

    # We count runs of equal values in D
    total_pairs = 0
    run_length = 1  # current run length

    for i in range(1, n-1):
        if D[i] == D[i-1]:
            run_length += 1
        else:
            # A run of length k in D contributes k*(k+1)//2 subarrays of length >= 2
            total_pairs += run_length * (run_length + 1) // 2
            run_length = 1

    # add the last run
    total_pairs += run_length * (run_length + 1) // 2

    # plus all single-element subarrays
    total_pairs += n

    print(total_pairs)

if __name__ == "__main__":
    main()
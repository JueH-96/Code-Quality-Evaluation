import sys
import math

def query(i, j):
    print(f"? {i} {j}")
    sys.stdout.flush()
    response = int(input())
    if response == -1:
        sys.exit()
    return response

def main():
    N, L, R = map(int, input().split())

    # Calculate the minimum number of questions required
    m = math.ceil(math.log2(R - L + 1))

    # Initialize the sum
    total_sum = 0

    # Query the judge for the sum of segments
    for i in range(m):
        j = L // (2 ** i)
        l = 2 ** i * j
        r = min(2 ** i * (j + 1) - 1, R)
        if l <= R:
            total_sum += query(i, j)
            total_sum %= 100

    # Print the final result
    print(f"! {total_sum}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
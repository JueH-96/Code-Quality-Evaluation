# YOUR CODE HERE
import sys

def query(i, j):
    print(f"? {i} {j}", flush=True)
    response = int(sys.stdin.readline().strip())
    if response == -1:
        sys.exit()
    return response

def main():
    import sys
    N, L, R = map(int, sys.stdin.readline().strip().split())
    total_length = 1 << N
    segment_sum = 0

    # Binary lifting to find the sum of the segment [L, R]
    for i in range(N, -1, -1):
        block_size = 1 << i
        if L % block_size == 0 and L + block_size - 1 <= R:
            segment_sum = (segment_sum + query(i, L // block_size)) % 100
            L += block_size

    print(f"! {segment_sum}", flush=True)

if __name__ == "__main__":
    main()
import sys

def main() -> None:
    # Read input fast
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, K = data[0], data[1]
    A = data[2:]
    
    # Sort the array
    A.sort()
    
    keep = N - K                     # number of elements that must remain
    best = 10 ** 18                  # sufficiently large
    
    # Slide a window of length `keep` over the sorted array
    for i in range(N - keep + 1):
        diff = A[i + keep - 1] - A[i]
        if diff < best:
            best = diff
    
    print(best)

if __name__ == "__main__":
    main()
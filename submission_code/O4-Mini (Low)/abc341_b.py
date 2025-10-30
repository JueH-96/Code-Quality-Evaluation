import sys
import threading

def main():
    import sys
    data = sys.stdin.read().strip().split()
    it = iter(data)
    n = int(next(it))
    # Read initial currencies
    A = [0] * (n + 1)
    for i in range(1, n + 1):
        A[i] = int(next(it))
    # Process exchanges
    for i in range(1, n):
        s = int(next(it))
        t = int(next(it))
        # Use as many exchanges as possible at country i
        k = A[i] // s
        A[i+1] += k * t
    # The answer is the amount at country N
    print(A[n])

if __name__ == "__main__":
    main()
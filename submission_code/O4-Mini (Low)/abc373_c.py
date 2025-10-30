import sys
import threading

def main():
    data = sys.stdin.read().split()
    # First token is N, then A_1 ... A_N, then B_1 ... B_N
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Read A values
    A_vals = [int(next(it)) for _ in range(n)]
    # Read B values
    B_vals = [int(next(it)) for _ in range(n)]
    # The maximum sum is simply max(A) + max(B)
    ans = max(A_vals) + max(B_vals)
    print(ans)

if __name__ == "__main__":
    main()
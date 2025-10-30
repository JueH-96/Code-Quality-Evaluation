#!/usr/bin/env python3
import sys

def main():
    N = int(sys.stdin.readline().strip())
    # For N >= 1, the answer is
    #   H = ceil(log2 N)
    #   f(N) = N*(H+1) - 2^H
    # It also works (and gives 0 for N=1) if we set
    #   H = (N-1).bit_length()
    H = (N-1).bit_length()
    result = N * (H + 1) - (1 << H)
    print(result)

if __name__ == "__main__":
    main()
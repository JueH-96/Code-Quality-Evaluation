import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    n = data[0]
    A = data[1:]

    NEG = -10**18          # sufficiently small negative value
    even = 0               # best score after processing current monsters with an even number defeated
    odd  = NEG             # best score with an odd  number defeated

    for x in A:
        # If we skip the monster, parity does not change (even, odd stay as they are)
        # If we defeat it:
        #   - coming from `even`   -> becomes odd, gain  x
        #   - coming from `odd`    -> becomes even, gain 2x
        next_even = max(even, odd + 2 * x)   # keep even, or defeat when we had odd
        next_odd  = max(odd,  even +     x)  # keep odd,  or defeat when we had even
        even, odd = next_even, next_odd

    print(max(even, odd))

if __name__ == "__main__":
    main()
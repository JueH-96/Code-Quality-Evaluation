import sys

def main() -> None:
    # Read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # The first token is N, but we actually don't need it since len(S) gives us that.
    # Still, we'll parse it to respect input format.
    _, S = data[0], data[1]
    
    seen = set()
    for idx, ch in enumerate(S, start=1):   # idx is 1-based
        seen.add(ch)
        if len(seen) == 3:                  # We have seen A, B, and C
            print(idx)
            return

if __name__ == "__main__":
    main()
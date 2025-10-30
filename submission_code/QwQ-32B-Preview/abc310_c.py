import sys

def main():
    lines = sys.stdin.read().splitlines()
    N = int(lines[0])
    sticks = lines[1:N+1]
    seen = set()
    for S in sticks:
        rev = S[::-1]
        canonical = min(S, rev)
        seen.add(canonical)
    print(len(seen))

if __name__ == "__main__":
    main()
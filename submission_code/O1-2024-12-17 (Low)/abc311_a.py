def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    seen = set()
    for i, ch in enumerate(S):
        seen.add(ch)
        if len(seen) == 3:  # A, B, and C all found
            print(i + 1)
            break

# Call main to execute solution
main()
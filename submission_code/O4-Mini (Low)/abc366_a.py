def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, T, A = map(int, data)
    remaining = N - T - A
    # If the current vote difference exceeds the remaining votes,
    # the leader cannot be overtaken.
    if abs(T - A) > remaining:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
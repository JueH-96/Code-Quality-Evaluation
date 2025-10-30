def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, T, A = map(int, input_data)
    # Remaining votes
    R = N - (T + A)
    # If the current vote difference is greater than remaining votes, outcome is decided
    if abs(T - A) > R:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
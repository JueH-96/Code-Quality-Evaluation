import sys

def main() -> None:
    S = sys.stdin.readline().rstrip('
')
    T = sys.stdin.readline().rstrip('
')

    # If the two strings are already equal, no operation is necessary.
    if S == T:
        print(0)
        return

    n = len(S)
    current = S
    answer = []

    # Each loop fixes exactly one mismatching position, choosing the one that
    # yields the lexicographically smallest possible next string.
    while current != T:
        best_string = None
        for i in range(n):
            if current[i] == T[i]:
                continue
            candidate = current[:i] + T[i] + current[i + 1:]
            if best_string is None or candidate < best_string:
                best_string = candidate

        # Record the chosen intermediate string and continue.
        answer.append(best_string)
        current = best_string

    # Output
    print(len(answer))
    for s in answer:
        print(s)

if __name__ == "__main__":
    main()
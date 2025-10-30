def main():
    import sys

    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Current string as a list for mutability
    curr = list(S)
    target = list(T)

    result = []

    # While current != target, perform one change per iteration
    while curr != target:
        best_str = None
        best_i = -1
        # For each position where they differ, try changing that char
        for i in range(len(curr)):
            if curr[i] == target[i]:
                continue
            # Form candidate by changing curr[i] to target[i]
            old = curr[i]
            curr[i] = target[i]
            cand = ''.join(curr)
            # Restore
            curr[i] = old
            # Keep lexicographically smallest candidate
            if best_str is None or cand < best_str:
                best_str = cand
                best_i = i
        # Apply the best change
        curr[best_i] = target[best_i]
        result.append(''.join(curr))

    # Output
    print(len(result))
    for s in result:
        print(s)

if __name__ == "__main__":
    main()
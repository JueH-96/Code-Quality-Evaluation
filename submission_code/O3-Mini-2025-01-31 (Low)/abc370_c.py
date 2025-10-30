def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    n = len(S)
    # The result array
    X = []
    
    # At any step S and T differ.
    # At each step we can change one character in S to its target value in T,
    # and our aim is to achieve minimal steps (which is exactly the number of differing positions)
    # and then among all valid sequences choose the lexicographically smallest sequence of strings.
    #
    # To get the lexicographically smallest overall sequence, it suffices to choose at each
    # step the move (changing one differing position) that gives the lexicographically smallest new S.
    # Because the total count of modifications is fixed (we fix each differing index exactly once),
    # the greedy approach of always taking the lexicographically smallest candidate string is correct.
    
    current = S
    while current != T:
        candidates = []
        # Consider each index where current differs from T,
        # and simulate changing that position to the target character.
        for i in range(n):
            if current[i] != T[i]:
                candidate = current[:i] + T[i] + current[i+1:]
                candidates.append(candidate)
        # Choose the lexicographically smallest candidate string.
        next_str = min(candidates)
        X.append(next_str)
        current = next_str

    # Output format: first output number of moves, then each modified S on its own line.
    output_lines = [str(len(X))]
    output_lines.extend(X)
    sys.stdout.write("
".join(output_lines))
    
if __name__ == "__main__":
    main()
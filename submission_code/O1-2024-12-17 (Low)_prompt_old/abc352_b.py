def solve():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    i, j = 0, 0  # Pointers for S and T
    positions = []
    while i < len(S) and j < len(T):
        if S[i] == T[j]:
            # If the current characters match, record position (1-based index)
            positions.append(j + 1)
            i += 1
        # Move to the next character in T regardless of match or mismatch
        j += 1
    
    print(" ".join(map(str, positions)))

def main():
    solve()

if __name__ == "__main__":
    main()
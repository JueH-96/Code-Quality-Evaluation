def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    Q = int(data[2])
    pairs = data[3:]  # c_i, d_i pairs follow

    # Mapping array M of size 26: M[i] = character that 'a'+i eventually maps to
    M = [chr(i + ord('a')) for i in range(26)]

    # Apply each operation in order
    idx = 0
    for _ in range(Q):
        c, d = pairs[idx], pairs[idx + 1]
        idx += 2
        # "All occurrences of c are replaced with d"
        # means: for each letter i, if it currently maps to c, make it map to d
        for i in range(26):
            if M[i] == c:
                M[i] = d

    # Construct the final string
    result = []
    for ch in S:
        result.append(M[ord(ch) - ord('a')])
    print("".join(result))

# Call solve() to read input, process, and print output
def main():
    solve()

if __name__ == "__main__":
    main()
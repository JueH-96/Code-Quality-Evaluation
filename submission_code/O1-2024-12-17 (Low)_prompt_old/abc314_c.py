def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = list(data[2])
    C = list(map(int, data[3:]))

    # For each color from 1..M, store all indices that have this color
    positions = [[] for _ in range(M+1)]
    for i in range(N):
        positions[C[i]].append(i)

    # Convert S to a list of chars for easy in-place modification
    for color in range(1, M+1):
        idx_list = positions[color]
        if len(idx_list) <= 1:
            # No effective shift needed if there's only one or zero characters of this color
            continue
        # Perform a right circular shift on these indices
        last_char = S[idx_list[-1]]
        for j in range(len(idx_list) - 1, 0, -1):
            S[idx_list[j]] = S[idx_list[j-1]]
        S[idx_list[0]] = last_char

    # Print the final string
    print("".join(S))

def main():
    solve()

if __name__ == "__main__":
    main()
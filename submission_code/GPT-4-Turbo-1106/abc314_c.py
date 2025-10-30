def right_circular_shift(s, indices):
    if not indices:
        return s
    last_char = s[indices[-1]]
    for i in range(len(indices) - 1, 0, -1):
        s[indices[i]] = s[indices[i - 1]]
    s[indices[0]] = last_char
    return s

def main():
    N, M = map(int, input().split())
    S = list(input().strip())
    C = list(map(int, input().split()))

    color_positions = {i: [] for i in range(1, M + 1)}
    for i, color in enumerate(C):
        color_positions[color].append(i)

    for i in range(1, M + 1):
        S = right_circular_shift(S, color_positions[i])

    print(''.join(S))

if __name__ == "__main__":
    main()
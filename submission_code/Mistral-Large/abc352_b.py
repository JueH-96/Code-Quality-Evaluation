import sys

def find_correct_positions(S, T):
    s_index = 0
    positions = []

    for t_index, t_char in enumerate(T):
        if s_index < len(S) and t_char == S[s_index]:
            positions.append(t_index + 1)
            s_index += 1

    return positions

def main():
    input = sys.stdin.read
    data = input().split()

    S = data[0]
    T = data[1]

    result = find_correct_positions(S, T)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()
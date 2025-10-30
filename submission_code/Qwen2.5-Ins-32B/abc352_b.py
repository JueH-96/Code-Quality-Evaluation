import sys

def find_correct_positions(S, T):
    s_index = 0
    correct_positions = []
    for t_index, char in enumerate(T):
        if s_index < len(S) and S[s_index] == char:
            correct_positions.append(t_index + 1)
            s_index += 1
    return correct_positions

if __name__ == "__main__":
    S = input().strip()
    T = input().strip()
    positions = find_correct_positions(S, T)
    print(" ".join(map(str, positions)))
def find_correct_positions(S, T):
    correct_positions = []
    for i in range(len(S)):
        if S[i] == T[i]:
            correct_positions.append(i+1)
    return correct_positions

S = input().strip()
T = input().strip()

correct_positions = find_correct_positions(S, T)

print(' '.join(map(str, correct_positions)))
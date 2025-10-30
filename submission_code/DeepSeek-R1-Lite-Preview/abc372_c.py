import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = list(input[ptr])
    ptr += 1
    abc_positions = set()
    for i in range(N - 2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            abc_positions.add(i)
    output = []
    for _ in range(Q):
        X_i = int(input[ptr])
        ptr += 1
        C_i = input[ptr]
        ptr += 1
        pos = X_i - 1  # Convert to 0-based index
        for delta in [-2, -1, 0]:
            i = pos + delta
            if 0 <= i <= N - 3:
                if i in abc_positions:
                    abc_positions.remove(i)
                if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
                    abc_positions.add(i)
        S[pos] = C_i
        output.append(str(len(abc_positions)))
    print('
'.join(output))

if __name__ == '__main__':
    main()
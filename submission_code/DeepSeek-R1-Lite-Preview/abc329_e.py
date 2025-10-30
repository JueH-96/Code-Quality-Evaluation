def can_transform(N, M, S, T):
    # Find all starting positions where S[i..i+M-1] == T
    operations = [i for i in range(N - M + 1) if S[i:i+M] == T]
    
    pos = 0
    while pos < N:
        # Find the earliest operation starting at or after pos
        # that covers pos
        op_index = -1
        for i in range(len(operations)):
            if operations[i] <= pos and operations[i] + M > pos:
                op_index = i
                break
        if op_index == -1:
            return "No"
        # Move pos to the end of this operation
        pos = operations[op_index] + M
    return "Yes"

# Read input
import sys
def main():
    import sys
    N_M = sys.stdin.readline().strip()
    while N_M == '':
        N_M = sys.stdin.readline().strip()
    N, M = map(int, N_M.split())
    S = sys.stdin.readline().strip()
    while len(S) < N:
        S += sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    while len(T) < M:
        T += sys.stdin.readline().strip()
    print(can_transform(N, M, S, T))

if __name__ == "__main__":
    main()
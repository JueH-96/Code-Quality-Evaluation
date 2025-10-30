# YOUR CODE HERE
import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    sequences = []
    
    def backtrack(current_sequence, index):
        if index == N:
            if current_sequence[-1] <= M:
                sequences.append(current_sequence.copy())
            return
        last = current_sequence[-1] if current_sequence else 0
        for a in range(last + 10, M + 1):
            current_sequence.append(a)
            backtrack(current_sequence, index + 1)
            current_sequence.pop()
    
    for a1 in range(1, M - 10*(N-1) + 1):
        backtrack([a1], 1)
    
    print(len(sequences))
    for seq in sorted(sequences):
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()
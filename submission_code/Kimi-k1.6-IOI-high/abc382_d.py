def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    
    result = []
    
    def backtrack(current_sequence):
        if len(current_sequence) == N:
            result.append(current_sequence.copy())
            return
        next_pos = len(current_sequence) + 1  # 1-based position
        if next_pos == 1:
            lower = 1
            upper = M - 10 * (N - 1)
        else:
            lower = current_sequence[-1] + 10
            upper = M - 10 * (N - next_pos)
        # Generate all possible values in ascending order
        for a in range(lower, upper + 1):
            current_sequence.append(a)
            backtrack(current_sequence)
            current_sequence.pop()
    
    backtrack([])
    print(len(result))
    for seq in result:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()
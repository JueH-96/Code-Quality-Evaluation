def main():
    import sys
    input = sys.stdin.readline
    S = input().strip()
    T = input().strip()
    
    correct_positions = []
    j = 0  # pointer for S
    for i, ch in enumerate(T):
        if j < len(S) and ch == S[j]:
            # this is the correctly typed character for S[j]
            correct_positions.append(i + 1)  # positions are 1-indexed
            j += 1
            if j == len(S):
                break
    print(" ".join(map(str, correct_positions)))

if __name__ == '__main__':
    main()
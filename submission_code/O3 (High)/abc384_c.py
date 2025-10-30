import sys
import itertools

def main() -> None:
    # Read the five problem scores
    a, b, c, d, e = map(int, sys.stdin.readline().split())
    scores = [a, b, c, d, e]
    letters = ['A', 'B', 'C', 'D', 'E']
    
    participants = []
    
    # Enumerate every non-empty subset of {A, B, C, D, E}
    # Masks 1 .. 31 (inclusive) correspond to the 2^5 - 1 = 31 subsets
    for mask in range(1, 32):
        name_chars = []
        total = 0
        for i in range(5):          # i = 0..4  ->  A..E
            if mask & (1 << i):
                name_chars.append(letters[i])
                total += scores[i]
        name = ''.join(name_chars)
        participants.append((name, total))
    
    # Sort: primary key = score descending, secondary key = name ascending
    participants.sort(key=lambda x: (-x[1], x[0]))
    
    # Output the names in order
    for name, _ in participants:
        print(name)

if __name__ == "__main__":
    main()
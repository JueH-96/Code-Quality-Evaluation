# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    C = list(map(int, data[3:3+N]))
    
    color_positions = [[] for _ in range(M+1)]
    for idx, c in enumerate(C):
        color_positions[c].append(idx)
    
    for i in range(1, M+1):
        positions = color_positions[i]
        k = len(positions)
        if k > 1:
            last_char = S[positions[-1]]
            # Shift right by 1
            for j in range(k-1, 0, -1):
                S[positions[j]] = S[positions[j-1]]
            S[positions[0]] = last_char
    
    print(''.join(S))

if __name__ == "__main__":
    main()
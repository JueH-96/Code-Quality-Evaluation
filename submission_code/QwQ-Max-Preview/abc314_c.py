from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    S = list(input[idx])
    idx += 1
    C = list(map(int, input[idx:idx + N]))
    idx += N
    
    # Initialize deques for each color (1-based)
    deques = [[] for _ in range(M + 1)]  # deques[0] unused
    for j in range(N):
        color = C[j]
        deques[color].append(j)
    
    # Convert lists to deques for efficient rotation
    for i in range(1, M + 1):
        deques[i] = deque(deques[i])
    
    current_string = S.copy()
    for i in range(1, M + 1):
        if len(deques[i]) <= 1:
            continue
        # Rotate the deque to the right by 1
        deques[i].rotate(1)
        positions = list(deques[i])
        # Collect the characters in the current positions
        chars = [current_string[p] for p in positions]
        # Rotate the characters right by 1
        if len(chars) >= 1:
            rotated_chars = [chars[-1]] + chars[:-1]
        else:
            rotated_chars = chars
        # Update the current_string with rotated characters
        for p in range(len(positions)):
            current_string[positions[p]] = rotated_chars[p]
    
    print(''.join(current_string))

if __name__ == "__main__":
    main()
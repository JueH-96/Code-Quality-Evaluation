def main():
    import sys
    S = sys.stdin.read().strip()
    N = len(S)
    if N < 3:
        print(0)
        return
    
    # Initialize left and right frequency arrays
    left = [0] * 26
    right = [0] * 26
    for char in S:
        right[ord(char) - ord('A')] += 1
    
    answer = 0
    for j in range(1, N + 1):
        current_char = S[j - 1]
        if 2 <= j <= N - 1:
            # Sum the product of frequencies in left and right
            for c in range(26):
                answer += left[c] * right[c]
        # Move current character from right to left
        left[ord(current_char) - ord('A')] += 1
        right[ord(current_char) - ord('A')] -= 1
    print(answer)

if __name__ == "__main__":
    main()
from collections import defaultdict

def main():
    import sys
    sys.setrecursionlimit(1000000)
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    freq = defaultdict(int)
    for c in S:
        freq[c] += 1
    total = 0

    def backtrack(state, freq, current_length):
        nonlocal total
        if current_length == N:
            total += 1
            return
        # Determine possible characters to add
        possible_chars = [c for c in freq if freq[c] > 0]
        for c in possible_chars:
            new_freq = freq.copy()
            new_freq[c] -= 1
            if current_length >= K - 1:
                new_substring = state + (c,)
                if len(new_substring) >= K:
                    is_pal = True
                    for i in range(K // 2):
                        if new_substring[i] != new_substring[K - 1 - i]:
                            is_pal = False
                            break
                    if is_pal:
                        continue  # Skip this character to avoid palindrome
            # Update state and proceed
            new_state = state[1:] + (c,)
            backtrack(new_state, new_freq, current_length + 1)

    backtrack((), freq, 0)
    print(total)

if __name__ == '__main__':
    main()
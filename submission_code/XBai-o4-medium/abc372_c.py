import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    s = list(sys.stdin.readline().strip())
    count = 0
    # Calculate initial count of "ABC" substrings
    for i in range(n - 2):
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            count += 1
    # Process each query
    for _ in range(q):
        x_str, c = sys.stdin.readline().split()
        x_idx = int(x_str) - 1  # convert to 0-based index
        # Subtract existing contributions from the three possible starting positions
        for s_candidate in [x_idx - 2, x_idx - 1, x_idx]:
            if 0 <= s_candidate <= len(s) - 3:
                if s[s_candidate] == 'A' and s[s_candidate + 1] == 'B' and s[s_candidate + 2] == 'C':
                    count -= 1
        # Update the character
        s[x_idx] = c
        # Add new contributions from the three possible starting positions
        for s_candidate in [x_idx - 2, x_idx - 1, x_idx]:
            if 0 <= s_candidate <= len(s) - 3:
                if s[s_candidate] == 'A' and s[s_candidate + 1] == 'B' and s[s_candidate + 2] == 'C':
                    count += 1
        print(count)

if __name__ == "__main__":
    main()
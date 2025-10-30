import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    s = list(input[idx])
    idx += 1

    total = 0
    for i in range(N - 2):
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            total += 1

    for _ in range(Q):
        x = int(input[idx]) - 1  # convert to 0-based
        idx += 1
        c = input[idx]
        idx += 1

        old_char = s[x]

        # Process old contributions
        starts = [x-2, x-1, x]
        for start in starts:
            if 0 <= start <= N - 3:
                if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
                    total -= 1

        # Update the character
        s[x] = c

        # Process new contributions
        for start in starts:
            if 0 <= start <= N - 3:
                if s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C':
                    total += 1

        print(total)

if __name__ == "__main__":
    main()
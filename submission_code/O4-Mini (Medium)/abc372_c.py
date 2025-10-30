import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    S = list(next(it).strip())
    # Count initial occurrences of "ABC"
    total = 0
    for i in range(N-2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            total += 1
    out = []
    for _ in range(Q):
        x = int(next(it)) - 1  # zero-based index
        c = next(it)
        # positions i where a "ABC" might start that includes x:
        # i in [x-2, x]
        for start in range(x-2, x+1):
            if 0 <= start <= N-3:
                if S[start] == 'A' and S[start+1] == 'B' and S[start+2] == 'C':
                    total -= 1
        # update character
        S[x] = c
        # re-count
        for start in range(x-2, x+1):
            if 0 <= start <= N-3:
                if S[start] == 'A' and S[start+1] == 'B' and S[start+2] == 'C':
                    total += 1
        out.append(str(total))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()
import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    S = data[0]
    T = data[1]
    n = len(S)
    res = [0] * n
    i = 0
    # Scan through T, match S as a subsequence
    for idx, ch in enumerate(T):
        if i < n and ch == S[i]:
            # record 1-based position
            res[i] = idx + 1
            i += 1
            if i == n:
                break
    # Output the recorded positions
    print(" ".join(map(str, res)))

if __name__ == "__main__":
    main()
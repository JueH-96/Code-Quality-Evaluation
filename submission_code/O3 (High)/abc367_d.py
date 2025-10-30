import sys
from collections import defaultdict

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, M = data[0], data[1]
    A = data[2:]

    # extended prefix‐sum values modulo M for two laps (indices 0 … 2N-1)
    ext_len = 2 * N
    ext = [0] * ext_len
    s = 0
    for i in range(1, ext_len):
        s += A[(i - 1) % N]
        ext[i] = s % M

    window = defaultdict(int)          # counts in the current window (i+1 … i+N-1)

    # initialise the first window: indices 1 … N-1
    for idx in range(1, N):
        window[ext[idx]] += 1

    ans = 0
    for i in range(N):                 # starting rest area index
        ans += window.get(ext[i], 0)   # add pairs (i, j) with ext[j] == ext[i]

        if i == N - 1:                 # last start – no need to shift the window
            break

        # slide the window one step forward:
        # 1. remove the element that goes out (i+1)
        rem = ext[i + 1]
        window[rem] -= 1
        if window[rem] == 0:
            del window[rem]

        # 2. add the new element that enters (i + N)
        window[ext[i + N]] += 1

    print(ans)

if __name__ == "__main__":
    main()
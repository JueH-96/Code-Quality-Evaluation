import sys
import threading

def main():
    S = sys.stdin.readline().strip()
    n = len(S)
    # cnt[c]: how many times char c has appeared so far
    # sumpos[c]: sum of positions (0-based) of char c seen so far
    cnt = [0] * 26
    sumpos = [0] * 26
    ans = 0

    for k, ch in enumerate(S):
        idx = ord(ch) - ord('A')
        c_cnt = cnt[idx]
        c_sum = sumpos[idx]
        # For each previous occurrence at position i, triples count contributions:
        # sum over i of (k - i - 1) = c_cnt * k - c_sum - c_cnt
        ans += c_cnt * k - c_sum - c_cnt
        cnt[idx] += 1
        sumpos[idx] += k

    print(ans)

if __name__ == "__main__":
    main()
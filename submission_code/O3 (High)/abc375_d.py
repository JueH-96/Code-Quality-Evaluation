import sys

def main() -> None:
    S = sys.stdin.readline().strip()

    # For each letter keep:
    #   counts[c]  – how many times the letter has appeared so far
    #   sums[c]    – sum of the (1-based) positions where it has appeared so far
    counts = [0] * 26
    sums   = [0] * 26

    ans = 0
    for pos, ch in enumerate(S, start=1):          # pos : 1-based position
        idx = ord(ch) - 65                         # 'A' -> 0 … 'Z' -> 25

        # New palindromic triples where current position is the right end:
        # For every previous occurrence p of this letter we get
        #     (p, j, pos)   with any j,  p < j < pos
        # Count contributed by all previous p:
        #     Σ (pos - p - 1) =
        #     (#previous)*(pos-1) - Σ p
        cnt  = counts[idx]
        ssum = sums[idx]
        ans += cnt * (pos - 1) - ssum

        # update bookkeeping for this letter
        counts[idx] = cnt + 1
        sums[idx]   = ssum + pos

    print(ans)

if __name__ == "__main__":
    main()
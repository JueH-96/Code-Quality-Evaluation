import sys

def main():
    S = sys.stdin.readline().strip()
    n = len(S)
    # Collect positions for each uppercase letter
    pos_lists = [[] for _ in range(26)]
    base = ord('A')
    for i, ch in enumerate(S):
        pos_lists[ord(ch) - base].append(i)
    # For each letter, count triples (i<j<k) with S[i]=S[k]=letter
    # For fixed letter, if its positions are p[0], p[1], ..., p[m-1],
    # the contribution is sum_{t=1..m-1} sum_{i=0..t-1} (p[t]-p[i]-1)
    # = sum_{t=1..m-1} [ t*(p[t]-1) - sum_{i=0..t-1} p[i] ].
    ans = 0
    for pos in pos_lists:
        m = len(pos)
        if m < 2:
            continue
        prefix_sum = 0
        for t, p in enumerate(pos):
            if t > 0:
                # Add t*(p-1) - sum_of_previous_positions
                ans += t * (p - 1) - prefix_sum
            prefix_sum += p
    print(ans)

if __name__ == "__main__":
    main()
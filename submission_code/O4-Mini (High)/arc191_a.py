import sys
def main():
    input = sys.stdin.readline
    # Read N, M and the two strings S and T
    N, M = map(int, input().split())
    S = input().strip()
    T = input().strip()
    # We must use T[M-1] exactly once in the final assignments
    ZERO = ord('0')
    last_val = ord(T[-1]) - ZERO      # T[M]
    # Count digits of T[1..M-1]
    counts = [0]*10
    for ch in T[:-1]:
        counts[ord(ch) - ZERO] += 1
    # cur_max = largest digit still available in counts
    cur_max = 9
    while cur_max > 0 and counts[cur_max] == 0:
        cur_max -= 1
    used_M = False
    res = []
    # Greedy over positions 1..N
    for i, ch in enumerate(S):
        dC = ord(ch) - ZERO  # original digit at S[i]
        # If this is the last position and we haven't used T[M], we must use it now
        if i == N-1 and not used_M:
            res.append(chr(last_val + ZERO))
            used_M = True
            continue
        # Option A: use the largest remaining T[k] for k < M (non‐M), if it strictly improves
        if cur_max > dC:
            dA = cur_max
        else:
            dA = -1
        # Option B: use T[M], if not used yet, and if it strictly improves
        if not used_M and last_val > dC:
            dB = last_val
        else:
            dB = -1
        # Decide which option gives the largest digit at this position
        # Tie‐break A vs B by preferring A (to save T[M] for later if equal)
        if dA >= dB and dA > dC:
            # take option A
            res.append(chr(dA + ZERO))
            counts[dA] -= 1
            # if we exhausted cur_max, move cur_max down
            if counts[dA] == 0 and dA == cur_max:
                while cur_max > 0 and counts[cur_max] == 0:
                    cur_max -= 1
        elif dB > dC:
            # take option B
            res.append(chr(dB + ZERO))
            used_M = True
        else:
            # take option C (keep original)
            res.append(chr(dC + ZERO))
    # Output the final string
    sys.stdout.write("".join(res))

# Call main
main()
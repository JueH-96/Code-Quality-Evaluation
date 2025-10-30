import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    S = list(input().rstrip('
'))
    Q = int(input())

    # For each position, store the time (operation index) when it was last assigned by t=1
    t_assign = [0] * N

    # Track the latest time of global lowercase (t=2) and uppercase (t=3) operations
    t_lower = 0
    t_upper = 0

    # Current operation index (1-based)
    current_time = 1

    for _ in range(Q):
        parts = input().split()
        ti = int(parts[0])
        if ti == 1:
            xi = int(parts[1]) - 1
            ci = parts[2]
            S[xi] = ci
            t_assign[xi] = current_time
        elif ti == 2:
            t_lower = current_time
        else:  # ti == 3
            t_upper = current_time
        current_time += 1

    # Build the final string
    res = []
    # We compare for each position: if its last individual assignment is more recent
    # than either global op, we keep it as is. Otherwise, we apply the later global op.
    latest_global = max(t_lower, t_upper)
    make_lower = t_lower > t_upper
    for i in range(N):
        if t_assign[i] > latest_global:
            # Individual assignment is more recent than any global operation
            res.append(S[i])
        else:
            # Global operation wins
            if make_lower:
                res.append(S[i].lower())
            else:
                res.append(S[i].upper())

    sys.stdout.write(''.join(res))

if __name__ == "__main__":
    main()
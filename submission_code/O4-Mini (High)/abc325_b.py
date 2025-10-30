def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    idx = 1
    # attend[t] will hold the total employees who can join
    # a meeting starting at UTC time t (lasting 1 hour).
    attend = [0] * 24
    for _ in range(n):
        w = int(data[idx]); x = int(data[idx+1])
        idx += 2
        # A base with offset x can attend if local start time
        # (x + T) mod 24 is between 9 and 17 inclusive.
        # For each local start k in [9..17], the UTC time is T = (k - x) mod 24.
        for local_start in range(9, 18):  # 9..17 inclusive
            t = (local_start - x) % 24
            attend[t] += w
    # The answer is the maximum over all possible UTC start times.
    print(max(attend))

main()
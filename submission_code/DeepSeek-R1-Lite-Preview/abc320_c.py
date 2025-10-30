def main():
    import sys
    M = int(sys.stdin.readline())
    S1 = sys.stdin.readline().strip()
    S2 = sys.stdin.readline().strip()
    S3 = sys.stdin.readline().strip()

    # Find common digits in S1, S2, and S3
    common_digits = set(S1) & set(S2) & set(S3)
    if not common_digits:
        print(-1)
        return

    T_max = 3 * M  # A safe upper bound for T
    min_T = float('inf')

    for c in common_digits:
        # For each reel, find times t where S_i[t % M] == c, up to T_max
        def get_times(S, c, M, T_max):
            positions = [i for i in range(M) if S[i] == c]
            times = []
            k = 0
            while True:
                for p in positions:
                    t = k * M + p
                    if t > T_max:
                        break
                    times.append(t)
                k += 1
                if k * M + positions[0] > T_max:
                    break
            return sorted(times)
        
        times1 = get_times(S1, c, M, T_max)
        times2 = get_times(S2, c, M, T_max)
        times3 = get_times(S3, c, M, T_max)
        
        # Use three-pointer approach to find the minimal T
        p1, p2, p3 = 0, 0, 0
        len1, len2, len3 = len(times1), len(times2), len(times3)
        
        while p1 < len1 and p2 < len2 and p3 < len3:
            t1, t2, t3 = times1[p1], times2[p2], times3[p3]
            current_T = max(t1, t2, t3)
            if t1 != t2 and t1 != t3 and t2 != t3:
                if current_T < min_T:
                    min_T = current_T
            # Move the pointer(s) with the smallest t
            min_t = min(t1, t2, t3)
            if t1 == min_t:
                p1 += 1
            if t2 == min_t:
                p2 += 1
            if t3 == min_t:
                p3 += 1

    if min_T == float('inf'):
        print(-1)
    else:
        print(min_T)

if __name__ == '__main__':
    main()
# YOUR CODE HERE
# Read inputs
M = int(input())
S = [input() for _ in range(3)]

INF = 10**9
min_t = INF

for c in '0123456789':
    # times_per_reel[i] will store the times when c appears on reel i
    times_per_reel = []
    for i in range(3):
        positions = [pos for pos, ch in enumerate(S[i]) if ch == c]
        times_i = set()
        if not positions:
            break  # c doesn't appear in this reel, so skip this c
        max_k = 10 * M  # Limit k to reasonable number
        for pos in positions:
            for k in range(10 * M):
                t_i = pos + M * k
                if t_i > min_t:
                    break  # No need to check times beyond current min_t
                times_i.add(t_i)
        times_per_reel.append(sorted(times_i))
    else:
        # All reels have c
        times_list = times_per_reel
        # Now, generate combinations
        for t0 in times_list[0]:
            if t0 >= min_t:
                break
            for t1 in times_list[1]:
                if t1 >= min_t:
                    break
                if t1 == t0:
                    continue
                for t2 in times_list[2]:
                    if t2 >= min_t:
                        break
                    if t2 == t0 or t2 == t1:
                        continue
                    t_current = max(t0, t1, t2)
                    if t_current < min_t:
                        min_t = t_current
                    break  # Since times are sorted, no need to check further t2
                # Early exit if t1 == t0
                #break
            # Early exit if t0 >= min_t
            #break

if min_t == INF:
    print(-1)
else:
    print(min_t)
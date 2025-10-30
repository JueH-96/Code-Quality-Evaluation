import sys
import threading

def main():
    import sys
    data = sys.stdin
    N_line = data.readline().strip()
    if not N_line:
        return
    N = int(N_line)
    # bucket[f] will collect all deliciousness S for flavor f
    # Flavor labels go up to N, so we can preallocate
    bucket = [[] for _ in range(N+1)]
    items = []  # list of (S, flavor)
    for _ in range(N):
        line = data.readline().split()
        f = int(line[0])
        s = int(line[1])
        bucket[f].append(s)
        items.append((s, f))
    # 1) compute the best same-flavor satisfaction
    max_same = 0
    for f in range(1, N+1):
        lst = bucket[f]
        if len(lst) < 2:
            continue
        # find the two largest S in lst
        m1 = 0
        m2 = 0
        for val in lst:
            if val > m1:
                m2 = m1
                m1 = val
            elif val > m2:
                m2 = val
        # same-flavor formula: m1 + m2/2  (note m2 is even so m2//2 is integer)
        cand = m1 + (m2 // 2)
        if cand > max_same:
            max_same = cand

    # 2) compute the best different-flavor satisfaction
    # sort all items descending by S
    items.sort(reverse=True, key=lambda x: x[0])
    max_diff = 0
    if len(items) >= 2:
        top_s, top_f = items[0]
        # find the first item with different flavor
        for s, f in items[1:]:
            if f != top_f:
                max_diff = top_s + s
                break

    # answer is the max of the two cases
    ans = max(max_same, max_diff)
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()
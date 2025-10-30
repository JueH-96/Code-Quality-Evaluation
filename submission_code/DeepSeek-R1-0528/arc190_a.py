import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    intervals = []
    index = 2
    for i in range(m):
        L = int(data[index]); R = int(data[index+1]); index += 2
        intervals.append((L, R))
    
    minR = 10**9
    maxL = 0
    for L, R in intervals:
        if R < minR:
            minR = R
        if L > maxL:
            maxL = L
    candidate2 = 2 if (minR + 1 <= maxL) else 10**9

    events = [[] for _ in range(n+2)]
    for i, (L, R) in enumerate(intervals):
        if L <= n:
            events[L].append((R, i))

    next_cover_all = [0] * (n+2)
    next_cover2 = [0] * (n+2)
    best_index = [-1] * (n+2)
    second_best_index = [-1] * (n+2)
    
    for x in range(1, n+1):
        next_cover_all[x] = next_cover_all[x-1]
        next_cover2[x] = next_cover2[x-1]
        best_index[x] = best_index[x-1]
        second_best_index[x] = second_best_index[x-1]
        
        for (R, idx) in events[x]:
            if R > next_cover_all[x]:
                next_cover2[x] = next_cover_all[x]
                second_best_index[x] = best_index[x]
                next_cover_all[x] = R
                best_index[x] = idx
            elif R == next_cover_all[x]:
                next_cover2[x] = R
                second_best_index[x] = idx
            elif R > next_cover2[x]:
                next_cover2[x] = R
                second_best_index[x] = idx

    steps0 = 0
    current = 0
    covering_intervals0 = []
    while current < n:
        x = current + 1
        if x > n:
            break
        next_val = next_cover_all[x]
        if next_val <= current:
            break
        if best_index[x] == -1:
            break
        covering_intervals0.append(best_index[x])
        current = next_val
        steps0 += 1
        
    if current < n:
        candidate0 = 10**9
    else:
        candidate0 = steps0

    candidate1 = 10**9
    best_i_candidate1 = -1
    best_covering = None

    for i in range(m):
        L_i, R_i = intervals[i]
        if L_i > R_i:
            continue
        steps_chain = 0
        current = L_i - 1
        covering_intervals_i = []
        valid = True
        while current < R_i:
            x = current + 1
            if x > R_i or x > n:
                break
            if best_index[x] == i:
                next_boundary = next_cover2[x]
                if next_boundary <= current:
                    valid = False
                    break
                if second_best_index[x] == -1:
                    valid = False
                    break
                covering_intervals_i.append(second_best_index[x])
                current = next_boundary
                steps_chain += 1
            else:
                next_boundary = next_cover_all[x]
                if next_boundary <= current:
                    valid = False
                    break
                if best_index[x] == -1:
                    valid = False
                    break
                covering_intervals_i.append(best_index[x])
                current = next_boundary
                steps_chain += 1
        if not valid or current < R_i:
            continue
        total_cost = 1 + steps_chain
        if total_cost < candidate1:
            candidate1 = total_cost
            best_i_candidate1 = i
            best_covering = covering_intervals_i

    total_min = min(candidate0, candidate1, candidate2)
    if total_min >= 10**9:
        print(-1)
        return

    ans_ops = ['0'] * m
    if total_min == candidate0:
        for idx in covering_intervals0:
            ans_ops[idx] = '1'
        print(candidate0)
        print(" ".join(ans_ops))
    elif total_min == candidate2:
        i0 = -1
        j0 = -1
        for i in range(m):
            if intervals[i][1] == minR:
                i0 = i
                break
        for i in range(m):
            if intervals[i][0] == maxL:
                j0 = i
                break
        ans_ops[i0] = '2'
        ans_ops[j0] = '2'
        print(2)
        print(" ".join(ans_ops))
    else:
        ans_ops[best_i_candidate1] = '2'
        for idx in best_covering:
            ans_ops[idx] = '1'
        print(candidate1)
        print(" ".join(ans_ops))

if __name__ == "__main__":
    main()
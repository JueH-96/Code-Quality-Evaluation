def main():
    from sys import stdin, stdout
    input = stdin.readline
    print = stdout.write

    N = int(input())
    segs = []
    for _ in range(N):
        L, R = map(int, input().split())
        segs.append((L, R))

    neg_sum, pos_sum = 0, 0
    neg_spots, pos_spots = [], []
    for L, R in segs:
        if L < 0:
            neg_sum += L
        if R > 0:
            pos_sum += R

    if (neg_sum + pos_sum) % 2 != 0 or neg_sum == 0 and pos_sum == 0:
        print('No')
        return

    for L, R in segs:
        if 0 <= L <= R:
            print('Yes
')
            return solve_0_case(N, segs)
        elif L < 0 <= R:
            return solve_mixed_case(N, segs)
        elif L < 0 and R < 0:
            neg_spots.append((L, R))
        elif L > 0 and R > 0:
            pos_spots.append((L, R))

    if pos_sum >= abs(neg_sum):
        for L, R in neg_spots:
            neg_sum += abs(L)
            print(f'{L} ')
        for L, R in pos_spots:
            to_take = min(R, abs(neg_sum))
            neg_sum += to_take
            print(f'{-to_take} ')
    else:
        for L, R in pos_spots:
            pos_sum -= R
            print(f'{R} ')
        for L, R in neg_spots:
            to_take = min(L, abs(pos_sum))
            pos_sum += to_take
            print(f'{to_take} ')

def solve_0_case(N, segs):
    neg_sum = 0
    pos_sum = 0
    pos_spots = []
    for i, (L, R) in enumerate(segs):
        if R < 0:
            neg_sum += R
            segs[i] = (R, R)
        elif L > 0:
            pos_sum += L
            pos_spots.append((L, L))
        else:
            if pos_sum < abs(neg_sum) and R > 0:
                to_add = min(R, abs(neg_sum) - pos_sum)
                segs[i] = (to_add, to_add)
                pos_sum += to_add
            elif pos_sum > abs(neg_sum) and L < 0:
                to_add = min(L, pos_sum + neg_sum)
                segs[i] = (to_add, to_add)
                neg_sum += to_add

    for L, R in pos_spots:
        segs.append((L, L))

    print(' '.join(map(str, get_ans(segs))))

def get_ans(segs):
    ans = [0] * len(segs)
    for i, (L, R) in enumerate(segs):
        ans[i] = R if neg_sum + pos_sum >= 0 else L

    neg_sum = sum(x for x in ans if x < 0)
    pos_sum = sum(x for x in ans if x > 0)

    if pos_sum > abs(neg_sum):
        for (i, x) in enumerate(ans):
            if x > 0:
                ans[i] -= pos_sum - abs(neg_sum)
                break

    return ans

def solve_mixed_case(N, segs):
    neg_sum, pos_sum = 0, 0
    for L, R in segs:
        if L < 0:
            neg_sum += L
        if R > 0:
            pos_sum += R

    if neg_sum <= pos_sum:
        adjust, direction = pos_sum - neg_sum, 1
    else:
        adjust, direction = neg_sum - pos_sum, -1

    for i, (L, R) in enumerate(segs):
        if L < 0 and R > 0:
            if direction == -1:
                to_add = min(abs(adjust), abs(L))
                adjust -= to_add
                segs[i] = (L - to_add, R)
            else:
                to_add = min(abs(adjust), R)
                adjust -= to_add
                segs[i] = (L, R - to_add)

    if adjust != 0:
        return "No"

    ans = get_ans(segs)
    print(' '.join(map(str, ans)))

main()
import sys

def main():
    n = int(sys.stdin.readline())
    pairs = []
    sum_L = 0
    sum_R = 0
    for _ in range(n):
        l, r = map(int, sys.stdin.readline().split())
        pairs.append((l, r))
        sum_L += l
        sum_R += r
    if sum_L > 0 or sum_R < 0:
        print("No")
        return
    x = [l for l, r in pairs]
    delta = -sum_L
    if delta == 0:
        print("Yes")
        print(' '.join(map(str, x)))
        return
    for i in range(n):
        l_i = x[i]
        r_i = pairs[i][1]
        possible_increase = r_i - l_i
        add = min(possible_increase, delta)
        x[i] += add
        delta -= add
        if delta == 0:
            break
    print("Yes")
    print(' '.join(map(str, x)))

if __name__ == "__main__":
    main()
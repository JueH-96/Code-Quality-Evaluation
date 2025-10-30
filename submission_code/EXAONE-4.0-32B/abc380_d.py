def main():
    S = input().strip()
    Q = int(input())
    K_list = list(map(int, input().split()))
    n0 = len(S)
    ans = []
    for K in K_list:
        total = n0
        flip = 0
        cur = K
        while total < cur:
            total *= 2
            half = total // 2
            if cur > half:
                cur -= half
                flip ^= 1
        while total > n0:
            half = total // 2
            if cur > half:
                cur -= half
            total = half
        char_at_cur = S[cur-1]
        if flip:
            if char_at_cur.islower():
                char_at_cur = char_at_cur.upper()
            else:
                char_at_cur = char_at_cur.lower()
        ans.append(char_at_cur)
    print(" ".join(ans))

if __name__ == '__main__':
    main()
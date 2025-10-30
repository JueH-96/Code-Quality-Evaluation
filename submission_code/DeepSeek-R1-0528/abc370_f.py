import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    total = sum(A)
    
    B = A + A
    P = [0] * (2*n+1)
    for i in range(1, 2*n+1):
        P[i] = P[i-1] + B[i-1]
    
    low, high = 0, total
    max_bits1 = (k-1).bit_length() if k-1 > 0 else 0
    x0 = 0
    next_seg_last = None

    while low <= high:
        mid = (low + high) // 2
        next_seg = [0] * (2*n+1)
        s = 0
        r = 0
        for l in range(2*n):
            if r < l:
                r = l
                s = 0
            while r < 2*n and s < mid:
                s += B[r]
                r += 1
            if s < mid:
                next_seg[l] = 2*n + 1
            else:
                next_seg[l] = r
            s -= B[l]
            if s < 0:
                s = 0

        if k == 1:
            if mid <= total:
                found = True
            else:
                found = False
        else:
            dp1 = [[0] * (2*n) for _ in range(max_bits1)]
            for i in range(2*n):
                if next_seg[i] < 2*n:
                    dp1[0][i] = next_seg[i]
                else:
                    dp1[0][i] = 2*n + 1

            for j in range(1, max_bits1):
                for i in range(2*n):
                    prev = dp1[j-1][i]
                    if prev < 2*n:
                        dp1[j][i] = dp1[j-1][prev]
                    else:
                        dp1[j][i] = 2*n + 1

            found = False
            for i in range(n):
                current = i
                rem = k - 1
                for bit in range(max_bits1):
                    if rem & (1 << bit):
                        if current >= 2*n:
                            break
                        current = dp1[bit][current]
                        if current > i + n:
                            break
                if current > i + n - 1:
                    continue
                seg_sum = P[i+n] - P[current]
                if seg_sum >= mid:
                    found = True
                    break

        if found:
            x0 = mid
            low = mid + 1
            next_seg_last = next_seg.copy()
        else:
            high = mid - 1

    if k == 1:
        next_seg_last = [0] * (2*n+1)
        s = 0
        r = 0
        for l in range(2*n):
            if r < l:
                r = l
                s = 0
            while r < 2*n and s < x0:
                s += B[r]
                r += 1
            if s < x0:
                next_seg_last[l] = 2*n + 1
            else:
                next_seg_last[l] = r
            s -= B[l]
            if s < 0:
                s = 0

    max_bits2 = (k).bit_length()
    dp2 = [[0] * (2*n) for _ in range(max_bits2)]
    for i in range(2*n):
        if next_seg_last[i] < 2*n:
            dp2[0][i] = next_seg_last[i]
        else:
            dp2[0][i] = 2*n + 1

    for j in range(1, max_bits2):
        for i in range(2*n):
            prev = dp2[j-1][i]
            if prev < 2*n:
                dp2[j][i] = dp2[j-1][prev]
            else:
                dp2[j][i] = 2*n + 1

    count_y = 0
    for i in range(n):
        s0 = i + 1
        s_end = s0 + n
        current = s0
        rem = k
        for bit in range(max_bits2):
            if rem & (1 << bit):
                if current >= 2*n:
                    break
                current = dp2[bit][current]
        if current != s_end:
            count_y += 1

    print(f"{x0} {count_y}")

if __name__ == '__main__':
    main()
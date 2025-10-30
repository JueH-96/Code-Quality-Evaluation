import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        X = int(input[ptr+1])
        K = int(input[ptr+2])
        ptr += 3
        if K == 0:
            print(1)
            continue
        ans = 0
        depth_of_X = X.bit_length() - 1
        max_a = min(K, depth_of_X)
        for a in range(0, max_a + 1):
            d = K - a
            if d < 0:
                continue
            current = X >> a
            if a == 0:
                if d > 60:
                    continue
                two_depth = 1 << d
                min_val = current * two_depth
                if min_val > N:
                    continue
                max_val = min_val + two_depth - 1
                if max_val <= N:
                    ans += two_depth
                else:
                    ans += N - min_val + 1
            else:
                if d == 0:
                    ans += 1
                else:
                    bit = (X >> (a - 1)) & 1
                    forbidden_child = current * 2 + bit
                    if forbidden_child == current * 2:
                        allowed_child = current * 2 + 1
                    else:
                        allowed_child = current * 2
                    if allowed_child > N:
                        continue
                    depth = d - 1
                    if depth > 60:
                        continue
                    two_depth = 1 << depth
                    min_val = allowed_child * two_depth
                    if min_val > N:
                        continue
                    max_val = min_val + two_depth - 1
                    if max_val <= N:
                        ans += two_depth
                    else:
                        ans += N - min_val + 1
        print(ans)

if __name__ == '__main__':
    main()
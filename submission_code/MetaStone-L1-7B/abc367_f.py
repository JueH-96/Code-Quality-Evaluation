import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N, Q = int(data[ptr]), int(data[ptr+1])
    ptr +=2
    A = list(map(int, data[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, data[ptr:ptr+N]))
    ptr +=N

    a_indices = defaultdict(list)
    for idx, x in enumerate(A):
        a_indices[x].append(idx +1)  # 1-based index

    b_indices = defaultdict(list)
    for idx, x in enumerate(B):
        b_indices[x].append(idx +1)  # 1-based index

    for _ in range(Q):
        l, r, L, R = map(int, data[ptr:ptr+4])
        ptr +=4

        len_a = r - l +1
        len_b = R - L +1
        if len_a != len_b:
            print("No")
            continue

        a_x = set()
        for x in A:
            if x not in a_indices:
                continue
            indices = a_indices[x]
            left = bisect.bisect_left(indices, l)
            if left < len(indices) and indices[left] <= r:
                a_x.add(x)

        b_x = set()
        for x in B:
            if x not in b_indices:
                continue
            indices = b_indices[x]
            left = bisect.bisect_left(indices, L)
            if left < len(indices) and indices[left] <= R:
                b_x.add(x)

        all_x = a_x.union(b_x)
        match = True

        for x in all_x:
            cnt_a = 0
            if x in a_indices:
                indices = a_indices[x]
                left = bisect.bisect_left(indices, l)
                right = bisect.bisect_right(indices, r)
                cnt_a = right - left
            else:
                cnt_a = 0

            cnt_b = 0
            if x in b_indices:
                indices = b_indices[x]
                left = bisect.bisect_left(indices, L)
                right = bisect.bisect_right(indices, R)
                cnt_b = right - left
            else:
                cnt_b = 0

            if cnt_a != cnt_b:
                match = False
                break

        print("Yes" if match else "No")

if __name__ == '__main__':
    main()
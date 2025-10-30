import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
    queries = [sys.stdin.readline().split() for _ in range(Q)]

    # Segment Tree Node definition
    class Node:
        __slots__ = ['a', 'b', 'left_B']
        def __init__(self, a=0, b=0, left_B=0):
            self.a = a  # max value if starting with add
            self.b = b  # max value if starting with mul
            self.left_B = left_B  # B of the leftmost element in this segment

    # Build the segment tree
    size = 1
    while size < N:
        size <<= 1
    tree = [Node() for _ in range(2 * size)]
    for i in range(N):
        tree[size + i].a = A[i]
        tree[size + i].b = 0
        tree[size + i].left_B = B[i]
    for i in range(size - 1, 0, -1):
        left = tree[2 * i]
        right = tree[2 * i + 1]
        if right.left_B == 0 and right.a == 0 and right.b == 0:
            tree[i].a = left.a
            tree[i].b = left.b
            tree[i].left_B = left.left_B
        else:
            B_right_first = right.left_B
            option1 = left.a + right.a
            option2 = left.a * B_right_first + right.a
            option3 = left.b * B_right_first + right.a
            tree[i].a = max(option1, option2, option3)

            option1 = left.a * right.left_B + right.b
            option2 = left.b * right.left_B + right.b
            tree[i].b = max(option1, option2)

            tree[i].left_B = left.left_B

    def update(pos):
        pos += size
        tree[pos].a = A[pos - size]
        tree[pos].b = 0
        tree[pos].left_B = B[pos - size]
        pos >>= 1
        while pos >= 1:
            left = tree[2 * pos]
            right = tree[2 * pos + 1]
            B_right_first = right.left_B if right.left_B != 0 else 0
            tree[pos].a = max(left.a + right.a, left.a * B_right_first + right.a, left.b * B_right_first + right.a)
            tree[pos].b = max(left.a * B_right_first + right.b, left.b * B_right_first + right.b)
            tree[pos].left_B = left.left_B
            pos >>= 1

    def query(l, r):
        res_a = 0
        res_b = 0
        l += size
        r += size
        left_segs = []
        right_segs = []
        while l <= r:
            if l % 2 == 1:
                left_segs.append(tree[l])
                l += 1
            if r % 2 == 0:
                right_segs.append(tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        segs = left_segs + right_segs[::-1]
        if not segs:
            return 0
        for seg in segs:
            B_right_first = seg.left_B
            new_a = max(res_a + seg.a, res_a * B_right_first + seg.a, res_b * B_right_first + seg.a)
            new_b = max(res_a * B_right_first + seg.b, res_b * B_right_first + seg.b)
            res_a, res_b = new_a, new_b
        return max(res_a, res_b)

    for query_t in queries:
        if query_t[0] == '1':
            i = int(query_t[1]) - 1
            x = int(query_t[2])
            A[i] = x
            update(i)
        elif query_t[0] == '2':
            i = int(query_t[1]) - 1
            x = int(query_t[2])
            B[i] = x
            update(i)
        else:
            l = int(query_t[1]) - 1
            r = int(query_t[2]) - 1
            print(query(l, r))

if __name__ == '__main__':
    main()
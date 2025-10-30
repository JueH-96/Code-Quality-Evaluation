import sys
import math

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1

    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    B = list(map(int, input[ptr:ptr+N]))
    ptr += N

    Q = int(input[ptr])
    ptr += 1

    class SegmentTree:
        def __init__(self, A, B):
            self.n = len(A)
            self.size = 1
            while self.size < self.n:
                self.size <<= 1
            self.add = [0] * (2 * self.size)
            self.mul = [0] * (2 * self.size)
            self.max_val = [0] * (2 * self.size)

            for i in range(self.n):
                self.add[self.size + i] = A[i]
                self.mul[self.size + i] = 0

            for i in range(self.size - 1, 0, -1):
                left = 2 * i
                right = 2 * i + 1
                a_left = self.add[left]
                m_left = self.mul[left]
                a_right = self.add[right]
                m_right = self.mul[right]

                self.add[i] = max(
                    a_left + a_right,
                    a_left + m_right,
                    m_left + a_right,
                    m_left + m_right
                )
                self.mul[i] = max(
                    a_left * a_right,
                    a_left * m_right,
                    m_left * a_right,
                    m_left * m_right
                )
                self.max_val[i] = max(self.add[i], self.mul[i])

        def update_pos(self, pos, new_val):
            pos += self.size
            self.add[pos] = new_val
            self.mul[pos] = 0

            pos >>= 1
            while pos >= 1:
                left = 2 * pos
                right = 2 * pos + 1
                a_left = self.add[left]
                m_left = self.mul[left]
                a_right = self.add[right]
                m_right = self.mul[right]

                new_add = max(
                    a_left + a_right,
                    a_left + m_right,
                    m_left + a_right,
                    m_left + m_right
                )
                new_mul = max(
                    a_left * a_right,
                    a_left * m_right,
                    m_left * a_right,
                    m_left * m_right
                )

                if self.add[pos] == new_add and self.mul[pos] == new_mul and self.max_val[pos] == max(new_add, new_mul):
                    break
                self.add[pos] = new_add
                self.mul[pos] = new_mul
                self.max_val[pos] = max(new_add, new_mul)
                pos >>= 1

        def query_range(self, l, r):
            res_add = -math.inf
            res_mul = -math.inf
            l += self.size
            r += self.size
            while l <= r:
                if l % 2 == 1:
                    res_add = max(res_add, self.add[l])
                    res_mul = max(res_mul, self.mul[l])
                    l += 1
                if r % 2 == 0:
                    res_add = max(res_add, self.add[r])
                    res_mul = max(res_mul, self.mul[r])
                    r -= 1
                l >>= 1
                r >>= 1
            return max(res_add, res_mul)

    st = SegmentTree(A, B)

    for _ in range(Q):
        query_type = int(input[ptr])
        ptr += 1
        l = int(input[ptr]) - 1  # convert to 0-based
        ptr += 1
        r = int(input[ptr]) - 1
        ptr += 1

        if query_type == 1 or query_type == 2:
            if query_type == 1:
                x = int(input[ptr])
                ptr += 1
                A[l] = x
            else:
                x = int(input[ptr])
                ptr += 1
                B[l] = x
            st.update_pos(l, A[l] if query_type == 1 else (A[l] * B[l] if B[l] != 0 else 0))
        else:
            max_val = st.query_range(l, r)
            print(max_val)

if __name__ == '__main__':
    main()
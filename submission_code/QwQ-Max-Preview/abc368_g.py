import sys
from sys import stdin
sys.setrecursionlimit(1 << 25)

def main():
    N = int(stdin.readline())
    A = list(map(int, stdin.readline().split()))
    B = list(map(int, stdin.readline().split()))
    Q = int(stdin.readline())
    queries = [stdin.readline().split() for _ in range(Q)]

    class SegmentTree:
        def __init__(self, dataA, dataB):
            self.n = len(dataA)
            self.size = 1
            while self.size < self.n:
                self.size <<= 1
            self.add = [0] * (2 * self.size)
            self.mul = [0] * (2 * self.size)
            for i in range(self.n):
                a = dataA[i]
                b_val = dataB[i]
                if i == 0:
                    self.add[self.size + i] = a
                    self.mul[self.size + i] = 0
                else:
                    self.add[self.size + i] = max(a, a * b_val)
                    self.mul[self.size + i] = max(a, a * b_val)
            for i in range(self.size - 1, 0, -1):
                left = 2 * i
                right = 2 * i + 1
                sa = self.add[left]
                ma = self.mul[left]
                sb = self.add[right]
                mb = self.mul[right]
                new_add = max(
                    sa + sb,
                    sa * mb + sb,
                    ma + sb,
                    ma * mb + sb
                )
                new_mul = max(sa * mb, ma * mb)
                self.add[i] = new_add
                self.mul[i] = new_mul

        def update(self, pos, a_val, b_val):
            pos += self.size
            if pos == 0:
                self.add[pos] = a_val
                self.mul[pos] = 0
            else:
                self.add[pos] = max(a_val, a_val * b_val)
                self.mul[pos] = max(a_val, a_val * b_val)
            pos >>= 1
            while pos >= 1:
                left = 2 * pos
                right = 2 * pos + 1
                sa = self.add[left]
                ma = self.mul[left]
                sb = self.add[right]
                mb = self.mul[right]
                new_add = max(sa + sb, sa * mb + sb, ma + sb, ma * mb + sb)
                new_mul = max(sa * mb, ma * mb)
                self.add[pos] = new_add
                self.mul[pos] = new_mul
                pos >>= 1

        def query(self, l, r):
            res_add = 0
            res_mul = 0
            l += self.size
            r += self.size
            stack = []
            stack.append((l, r, False))
            while stack:
                a, b, pushed = stack.pop()
                if a > b:
                    continue
                if not pushed:
                    stack.append((a, b, True))
                    if a == b:
                        left_add = self.add[a]
                        left_mul = self.mul[a]
                        res_add = left_add
                        res_mul = left_mul
                        continue
                    stack.append((a, (a + b) // 2, False))
                    stack.append(((a + b) // 2 + 1, b, False))
                else:
                    left_add = res_add
                    left_mul = res_mul
                    right_add = self.add[b]
                    right_mul = self.mul[b]
                    if a == b:
                        res_add = left_add
                        res_mul = left_mul
                    else:
                        new_add = max(
                            left_add + right_add,
                            left_add * right_mul + right_add,
                            left_mul + right_add,
                            left_mul * right_mul + right_add
                        )
                        new_mul = max(left_add * right_mul, left_mul * right_mul)
                        res_add = new_add
                        res_mul = new_mul
            return (res_add, res_mul)

    st = SegmentTree(A, B)

    for query in queries:
        if query[0] == '1':
            i = int(query[1]) - 1
            x = int(query[2])
            A[i] = x
            st.update(i, x, B[i])
        elif query[0] == '2':
            i = int(query[1]) - 1
            x = int(query[2])
            B[i] = x
            st.update(i, A[i], x)
        else:
            l = int(query[1]) - 1
            r = int(query[2]) - 1
            if l == r:
                print(A[l])
                continue
            a = A[l]
            add_val, mul_val = st.query(l + 1, r)
            option1 = a + add_val
            option2 = a * mul_val
            print(max(option1, option2))

if __name__ == "__main__":
    main()
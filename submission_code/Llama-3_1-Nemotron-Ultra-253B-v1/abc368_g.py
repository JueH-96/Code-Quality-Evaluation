import sys
import math

CAP = 10**18

class SumSegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
    
    def update(self, pos, value):
        pos += self.size
        self.tree[pos] = value
        pos >>= 1
        while pos >= 1:
            new_val = self.tree[2 * pos] + self.tree[2 * pos + 1]
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
            pos >>= 1
    
    def query(self, l, r):
        res = 0
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res += self.tree[l]
                l += 1
            if r % 2 == 0:
                res += self.tree[r]
                r -= 1
            l >>= 1
            r >>= 1
        return res

class ProductSegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [1] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            left = self.tree[2 * i]
            right = self.tree[2 * i + 1]
            if left >= CAP or right >= CAP:
                self.tree[i] = CAP
            else:
                product = left * right
                self.tree[i] = product if product <= CAP else CAP
    
    def update(self, pos, value):
        pos += self.size
        self.tree[pos] = value
        pos >>= 1
        while pos >= 1:
            left = self.tree[2 * pos]
            right = self.tree[2 * pos + 1]
            if left >= CAP or right >= CAP:
                new_val = CAP
            else:
                new_val = left * right
                if new_val > CAP:
                    new_val = CAP
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
            pos >>= 1
    
    def query(self, l, r):
        res = 1
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                if res >= CAP or self.tree[l] >= CAP:
                    res = CAP
                else:
                    res *= self.tree[l]
                    if res > CAP:
                        res = CAP
                l += 1
            if r % 2 == 0:
                if res >= CAP or self.tree[r] >= CAP:
                    res = CAP
                else:
                    res *= self.tree[r]
                    if res > CAP:
                        res = CAP
                r -= 1
            l >>= 1
            r >>= 1
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    B = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    sum_tree = SumSegmentTree(A)
    product_tree = ProductSegmentTree(B)
    Q = int(input[ptr])
    ptr +=1
    for _ in range(Q):
        query = input[ptr]
        ptr +=1
        if query == '1':
            i = int(input[ptr]) -1
            ptr +=1
            x = int(input[ptr])
            ptr +=1
            sum_tree.update(i, x)
        elif query == '2':
            i = int(input[ptr]) -1
            ptr +=1
            x = int(input[ptr])
            ptr +=1
            product_tree.update(i, x)
        else:
            l = int(input[ptr])
            ptr +=1
            r = int(input[ptr])
            ptr +=1
            l_0 = l -1
            r_0 = r -1
            low = l_0
            high = r_0
            ans = l_0
            while low <= high:
                mid = (low + high) // 2
                current_val = 0
                if mid >= l_0:
                    sum_A = sum_tree.query(l_0, mid)
                    product_B = product_tree.query(mid +1, r_0) if (mid +1 <= r_0) else 1
                    current_val = sum_A * product_B
                else:
                    current_val = 0 * product_tree.query(l_0, r_0)
                next_val = 0
                if mid +1 <= r_0:
                    sum_A_next = sum_tree.query(l_0, mid +1)
                    product_B_next = product_tree.query(mid +2, r_0) if (mid +2 <= r_0) else 1
                    next_val = sum_A_next * product_B_next
                else:
                    next_val = sum_tree.query(l_0, r_0) * 1
                if current_val < next_val:
                    ans = mid +1
                    low = mid +1
                else:
                    high = mid -1
            max_val = 0
            if ans >= l_0:
                sum_A = sum_tree.query(l_0, ans)
                product_B = product_tree.query(ans +1, r_0) if (ans +1 <= r_0) else 1
                max_val = sum_A * product_B
            else:
                max_val = 0 * product_tree.query(l_0, r_0)
            if ans < r_0:
                sum_A_r = sum_tree.query(l_0, r_0)
                product_B_r = 1
                candidate = sum_A_r * product_B_r
                if candidate > max_val:
                    max_val = candidate
            print(max_val)

if __name__ == '__main__':
    main()
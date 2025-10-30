import heapq

class BIT:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)

    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    events = []
    for _ in range(M):
        T_i = int(input[ptr])
        ptr += 1
        W_i = int(input[ptr])
        ptr += 1
        S_i = int(input[ptr])
        ptr += 1
        events.append((T_i, W_i, S_i))

    available_data = [0] * (N + 1)
    bit = BIT(N)
    for i in range(1, N + 1):
        available_data[i] = 1
        bit.update(i, 1)

    return_heap = []
    return_times = dict()
    ans = [0] * (N + 1)

    for T_i, W_i, S_i in events:
        while return_heap and return_heap[0][0] <= T_i:
            return_time, person = heapq.heappop(return_heap)
            if person in return_times and return_times[person] == return_time:
                if available_data[person] == 0:
                    available_data[person] = 1
                    bit.update(person, 1)
                del return_times[person]

        low = 1
        high = N
        res = -1
        while low <= high:
            mid = (low + high) // 2
            sum_mid = bit.query(mid)
            if sum_mid == 0:
                low = mid + 1
            else:
                sum_mid_minus_1 = bit.query(mid - 1) if mid > 1 else 0
                if sum_mid_minus_1 == 0 and sum_mid >= 1:
                    res = mid
                    break
                elif sum_mid_minus_1 >= 1:
                    high = mid - 1
                else:
                    low = mid + 1

        if res != -1:
            ans[res] += W_i
            available_data[res] = 0
            bit.update(res, -1)
            new_return_time = T_i + S_i
            if res in return_times:
                old_return_time = return_times[res]
                if new_return_time < old_return_time:
                    return_times[res] = new_return_time
            else:
                return_times[res] = new_return_time
            heapq.heappush(return_heap, (new_return_time, res))

    for i in range(1, N + 1):
        print(ans[i])

if __name__ == '__main__':
    main()
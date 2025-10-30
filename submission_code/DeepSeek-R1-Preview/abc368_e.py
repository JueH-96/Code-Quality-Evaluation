import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    X1 = int(data[idx])
    idx += 1

    trains = []
    original_index = []
    for i in range(M):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        S = int(data[idx])
        idx += 1
        T = int(data[idx])
        idx += 1
        trains.append((S, A, B, T, i))
        original_index.append(i)

    trains.sort()
    sorted_indices = [t[4] for t in trains]
    X = [0] * M
    city_data = dict()

    for idx_in_sorted, train in enumerate(trains):
        S_j, A_j, B_j, T_j, original_idx = train
        if original_idx == 0:
            X_j = X1
        else:
            c = A_j
            if c not in city_data:
                max_val = 0
            else:
                t_list = city_data[c]['times']
                max_bit = city_data[c]['max_bit']
                pos = bisect.bisect_right(t_list, S_j)
                if pos == 0:
                    max_val = 0
                else:
                    max_val = max_bit.query(pos)
            X_j = max(0, max_val - S_j)
        X[original_idx] = X_j
        arrival_time = T_j + X_j
        if B_j not in city_data:
            times = []
            values = []
            city_data[B_j] = {
                'times': times,
                'values': values,
                'max_bit': None
            }
        else:
            times = city_data[B_j]['times']
            values = city_data[B_j]['values']
        pos = bisect.bisect_left(times, arrival_time)
        times.insert(pos, arrival_time)
        values.insert(pos, arrival_time + X_j)
        n = len(times)
        max_bit = FenwickTree(n)
        for i in range(n):
            max_bit.update(i + 1, values[i])
        city_data[B_j]['max_bit'] = max_bit

    output = []
    for i in range(M):
        if i == 0:
            continue
        output.append(str(X[i]))
    print(' '.join(output))

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)

    def update(self, idx, value):
        while idx <= self.n:
            if self.tree[idx] < value:
                self.tree[idx] = value
            else:
                break
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res = max(res, self.tree[idx])
            idx -= idx & -idx
        return res

if __name__ == '__main__':
    main()
import sys
import bisect

class FenwTreeMax:
    def __init__(self, size):
        self.n = size
        self.tree = [-10**18] * (self.n + 1)
    
    def update(self, index, val):
        while index <= self.n:
            if val > self.tree[index]:
                self.tree[index] = val
            index += index & -index
            
    def query(self, index):
        res = -10**18
        while index:
            if self.tree[index] > res:
                res = self.tree[index]
            index -= index & -index
        return res

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    X1 = int(next(it))
    
    trains = []
    for i in range(m):
        A = int(next(it))
        B = int(next(it))
        S = int(next(it))
        T = int(next(it))
        trains.append((A, B, S, T, i + 1))
    
    arrivals_per_city = {}
    for A, B, S, T, idx in trains:
        if B not in arrivals_per_city:
            arrivals_per_city[B] = []
        arrivals_per_city[B].append(T)
    
    trees = {}
    for city, times in arrivals_per_city.items():
        arr = sorted(set(times))
        mapping = {t: idx + 1 for idx, t in enumerate(arr)}
        fenw_tree = FenwTreeMax(len(arr))
        trees[city] = (mapping, arr, fenw_tree)
    
    res = [0] * (m + 1)
    res[1] = X1
    
    sorted_trains = sorted(trains, key=lambda x: x[2])
    
    for train in sorted_trains:
        A, B, S, T, idx = train
        if idx == 1:
            x = res[1]
        else:
            if A in trees:
                mapping, arr, tree_obj = trees[A]
                pos = bisect.bisect_right(arr, S) - 1
                if pos < 0:
                    V_max = -10**18
                else:
                    V_max = tree_obj.query(pos + 1)
            else:
                V_max = -10**18
            x = max(0, V_max - S)
            res[idx] = x
        
        if B in trees:
            mapping, arr, tree_obj = trees[B]
            if T in mapping:
                pos_index = mapping[T]
                value = x + T
                tree_obj.update(pos_index, value)
    
    output = []
    for i in range(2, m + 1):
        output.append(str(res[i]))
    print(" ".join(output))

if __name__ == '__main__':
    main()
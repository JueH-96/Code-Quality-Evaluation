import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    X1 = int(input[ptr]); ptr +=1

    trains = []
    for i in range(M):
        A = int(input[ptr]); ptr +=1
        B = int(input[ptr]); ptr +=1
        S = int(input[ptr]); ptr +=1
        T = int(input[ptr]); ptr +=1
        trains.append( (A, B, S, T, i) )

    # Separate the first train (index 0)
    first_A, first_B, first_S, first_T, first_idx = trains[0]
    X = [0] * M
    X[first_idx] = X1

    from collections import defaultdict
    arrival = defaultdict(list)
    departure = defaultdict(list)

    for i in range(1, M):
        A, B, S, T, idx = trains[i]
        departure[A].append( (S, B, T, idx) )
        arrival[B].append( (T, idx) )

    # Sort arrival lists by T
    for city in arrival:
        arrival[city].sort()

    # Sort departure lists by S
    for city in departure:
        departure[city].sort()

    # Fenwick Tree for maximum query
    class FenwickTree:
        def __init__(self):
            self.tree = []
            self.size = 1
            while self.size < len(arrival):
                self.size <<=1
            self.tree = [0] * (2 * self.size)

        def update(self, pos, value):
            pos += self.size
            while pos >=1:
                if self.tree[pos] < value:
                    self.tree[pos] = value
                else:
                    break
                pos >>=1

        def query(self, pos):
            res = 0
            pos += self.size
            while pos >=1:
                if self.tree[pos] > res:
                    res = self.tree[pos]
                pos >>=1
            return res

    fenwick = defaultdict(FenwickTree)

    # Insert the first train into its arrival city
    ft_first_city = fenwick[first_B]
    pos = bisect.bisect_right(arrival[first_B], (first_T, first_idx)) 
    if pos >0:
        ft_first_city.update(pos-1, X1 + first_T)

    # Process other departure trains in order of S
    dep_sorted = []
    for city in departure:
        for (s, B, t, idx) in departure[city]:
            dep_sorted.append( (s, city, B, t, idx) )
    dep_sorted.sort()

    # Initialize X for others
    for s, u, B, t, idx in dep_sorted:
        ft = fenwick.get(u, FenwickTree())
        arr_list = arrival.get(u, [])
        if not arr_list:
            max_val = 0
        else:
            # Find the largest T_i <= s
            left = 0
            right = len(arr_list) -1
            best = -1
            while left <= right:
                mid = (left + right) //2
                if arr_list[mid][0] <= s:
                    best = mid
                    left = mid +1
                else:
                    right = mid -1
            if best == -1:
                max_val =0
            else:
                max_val = ft.query(best)
        X[idx] = max(0, max_val - s)
        # Insert into B's arrival
        ft_B = fenwick[B]
        pos = bisect.bisect_right(arrival[B], (t, idx)) 
        if pos >0:
            ft_B.update(pos-1, X[idx] + t)
        elif pos == len(arrival[B]):
            ft_B.update(len(arrival[B]), X[idx] + t)

    # Output the X's in order, excluding the first train
    output = []
    for i in range(1, M):
        output.append(str(X[i]))
    print(' '.join(output))

if __name__ == '__main__':
    main()
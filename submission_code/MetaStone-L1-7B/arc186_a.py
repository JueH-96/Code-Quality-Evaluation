import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(Q):
        queries.append(int(input[ptr]))
        ptr += 1

    possible = set()

    # Function to compute fixed count for given s and t
    def compute_fixed(s, t):
        a = s.count(0)
        b = s.count(N)
        c = t.count(0)
        d = t.count(N)
        fixed = (a * N + b * N) + (c * N + d * N)
        fixed -= (a * c + a * d + b * c + b * d)
        e = 0  # Placeholder for cells where s_i=1 and t_j=1
        fixed += e
        return fixed

    # Iterate over all possible row sums and column sums
    for s in range(0, N+1):
        for t in range(0, N+1):
            if s != t:
                continue
            # Generate all possible row sum vectors with sum s
            from itertools import combinations
            rows = []
            for comb in combinations(range(N), s):
                row = [1 if i in comb else 0 for i in range(N)]
                rows.append(row)
            # Similarly for columns
            cols = []
            for comb in combinations(range(N), t):
                col = [1 if i in comb else 0 for i in range(N)]
                cols.append(col)
            # Compute fixed counts
            for row in rows:
                for col in cols:
                    fixed = 0
                    for i in range(N):
                        if row[i] == 0:
                            fixed += 1
                        elif row[i] == N:
                            fixed += 1
                        if col[i] == 0:
                            fixed += 1
                        elif col[i] == N:
                            fixed += 1
                        if row[i] == 1 and col[i] == 1:
                            fixed += 1
                    possible.add(fixed)

    # Now, for each query, check if it's in possible
    for K in queries:
        if K == 0:
            if 0 in possible:
                print("Yes")
            else:
                print("No")
        elif K == N * N:
            if K in possible:
                print("Yes")
            else:
                print("No")
        else:
            print("No")

if __name__ == "__main__":
    main()
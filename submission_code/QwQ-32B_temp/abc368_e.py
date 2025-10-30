import sys
import bisect
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N, M, X1_val = map(int, sys.stdin.readline().split())
    trains = []
    for idx in range(1, M+1):
        A, B, S, T = map(int, sys.stdin.readline().split())
        trains.append((S, A, B, T, idx))
    
    # Sort trains by their departure time S
    trains.sort(key=lambda x: x[0])
    
    # For each city, maintain entries and T_list
    city_entries = defaultdict(lambda: {'entries': [], 'T_list': []})
    X = [0] * (M + 1)  # X[1..M]
    
    for train in trains:
        S, A, B, T, idx = train
        if idx == 1:
            # First train
            X_val = X1_val
            val = X_val + T
            entries = city_entries[B]['entries']
            T_list = city_entries[B]['T_list']
            while entries and entries[-1][0] <= T:
                if val > entries[-1][1]:
                    entries.pop()
                    T_list.pop()
                else:
                    break
            if not entries or entries[-1][0] < T:
                entries.append((T, val))
                T_list.append(T)
            else:
                if val > entries[-1][1]:
                    entries[-1] = (T, val)
                    T_list[-1] = T
            X[idx] = X_val
        else:
            # Find max_val in city A's entries where T_i <= S
            entries_A = city_entries[A]['entries']
            T_list_A = city_entries[A]['T_list']
            max_val = -float('inf')
            if T_list_A:
                idx_T = bisect.bisect_right(T_list_A, S) - 1
                if idx_T >= 0:
                    max_val = entries_A[idx_T][1]
            candidate = max_val - S
            X_val = max(candidate, 0)
            val = X_val + T
            # Update city B's entries and T_list
            entries_B = city_entries[B]['entries']
            T_list_B = city_entries[B]['T_list']
            while entries_B and entries_B[-1][0] <= T:
                if val > entries_B[-1][1]:
                    entries_B.pop()
                    T_list_B.pop()
                else:
                    break
            if not entries_B or entries_B[-1][0] < T:
                entries_B.append((T, val))
                T_list_B.append(T)
            else:
                if val > entries_B[-1][1]:
                    entries_B[-1] = (T, val)
                    T_list_B[-1] = T
            X[idx] = X_val
    
    # Collect results for trains 2 to M in order
    result = []
    for idx in range(2, M+1):
        result.append(str(X[idx]))
    print(' '.join(result))

if __name__ == "__main__":
    main()
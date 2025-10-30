import sys
import bisect

def find_insert_pos_descending(arr, x):
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] > x:
            lo = mid + 1
        else:
            hi = mid
    return lo

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    adj_list = [[] for _ in range(N+1)]
    adj_set = [set() for _ in range(N+1)]

    for v in range(1, N+1):
        adj_list[v].append(v)
        adj_set[v].add(v)

    for _ in range(Q):
        query = input[ptr]
        ptr += 1
        if query == '1':
            u = int(input[ptr])
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            if v not in adj_set[u]:
                pos = find_insert_pos_descending(adj_list[u], v)
                adj_list[u].insert(pos, v)
                adj_set[u].add(v)
            if u not in adj_set[v]:
                pos = find_insert_pos_descending(adj_list[v], u)
                adj_list[v].insert(pos, u)
                adj_set[v].add(u)
        elif query == '2':
            v = int(input[ptr])
            ptr += 1
            k = int(input[ptr])
            ptr += 1
            if len(adj_list[v]) >= k:
                print(adj_list[v][k-1])
            else:
                print(-1)

if __name__ == '__main__':
    main()
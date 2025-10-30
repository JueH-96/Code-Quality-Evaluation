import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    q = int(data[1])
    queries = list(map(int, data[2:2+q]))
    
    in_set = [False] * (n + 1)
    toggle_times = [[] for _ in range(n + 1)]
    T_arr = [0] * q
    cur_size = 0
    
    for idx, x in enumerate(queries):
        if in_set[x]:
            cur_size -= 1
            in_set[x] = False
        else:
            cur_size += 1
            in_set[x] = True
        T_arr[idx] = cur_size
        toggle_times[x].append(idx)
    
    P = [0] * (q + 1)
    for i in range(1, q + 1):
        P[i] = P[i - 1] + T_arr[i - 1]
    
    res = [0] * (n + 1)
    
    for j in range(1, n + 1):
        lst = toggle_times[j]
        m = len(lst)
        total_contrib = 0
        for i in range(0, m, 2):
            start = lst[i]
            if i + 1 < m:
                end = lst[i + 1] - 1
            else:
                end = q - 1
            if start <= end:
                total_contrib += P[end + 1] - P[start]
        res[j] = total_contrib
        
    print(" ".join(str(x) for x in res[1:]))

if __name__ == "__main__":
    main()
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    q = int(data[1])
    xs = list(map(int, data[2:2+q]))
    
    events = [[] for _ in range(n+1)]
    for i in range(q):
        x = xs[i]
        events[x].append(i+1)
        
    in_set = [False] * (n+1)
    current_size = 0
    F_arr = [0] * q
    
    for i in range(q):
        x = xs[i]
        if in_set[x]:
            current_size -= 1
            in_set[x] = False
        else:
            current_size += 1
            in_set[x] = True
        F_arr[i] = current_size
        
    P = [0] * (q+1)
    for i in range(1, q+1):
        P[i] = P[i-1] + F_arr[i-1]
        
    ans = [0] * (n+1)
    for j in range(1, n+1):
        toggles = events[j]
        m = len(toggles)
        total_val = 0
        k = 0
        while k < m:
            start = toggles[k]
            if k + 1 < m:
                end = toggles[k+1] - 1
                k += 2
            else:
                end = q
                k += 1
            total_val += P[end] - P[start-1]
        ans[j] = total_val
        
    res = []
    for j in range(1, n+1):
        res.append(str(ans[j]))
    print(" ".join(res))

if __name__ == "__main__":
    main()
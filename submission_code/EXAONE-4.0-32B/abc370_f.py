import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    total = sum(A)
    max_possible_x = total // k
    
    def compute_nxt(x_val):
        B = A + A
        N2 = len(B)
        nxt_arr = [10**18] * N2
        cur_sum = 0
        r = 0
        for l in range(N2):
            if r < l:
                r = l
                cur_sum = 0
            while r < min(l + n, N2) and cur_sum < x_val:
                cur_sum += B[r]
                r += 1
            if cur_sum >= x_val:
                nxt_arr[l] = r - 1
            else:
                nxt_arr[l] = 10**18
            if r > l:
                cur_sum -= B[l]
            else:
                r = l + 1
        return nxt_arr

    def feasible(x_val):
        if x_val == 0:
            return True
        nxt_arr = compute_nxt(x_val)
        N2 = 2 * n
        dp_circle = [0] * (N2 + 1)
        for i in range(N2 - 1, -1, -1):
            if nxt_arr[i] == 10**18:
                dp_circle[i] = 0
            else:
                next_start = nxt_arr[i] + 1
                if next_start > i + n - 1:
                    dp_circle[i] = 1
                else:
                    dp_circle[i] = 1 + dp_circle[next_start]
        for i in range(n):
            if dp_circle[i] >= k:
                return True
        return False

    low, high = 0, max_possible_x + 1
    while low < high:
        mid = (low + high + 1) // 2
        if feasible(mid):
            low = mid
        else:
            high = mid - 1
    x_final = low

    nxt_arr = compute_nxt(x_final)
    N2 = 2 * n
    g = [0] * (N2 + 1)
    for i in range(N2 - 1, -1, -1):
        if nxt_arr[i] == 10**18:
            g[i] = 0
        else:
            next_start = nxt_arr[i] + 1
            if next_start > i + n - 1:
                g[i] = 1
            else:
                g[i] = 1 + g[next_start]
                
    y_count = 0
    for i in range(n):
        if g[i + 1] < k:
            y_count += 1
            
    print(f"{x_final} {y_count}")

if __name__ == "__main__":
    main()
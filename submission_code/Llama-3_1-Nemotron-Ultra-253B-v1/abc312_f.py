import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    T0 = []
    T1 = []
    T2 = []
    for _ in range(N):
        T, X = map(int, sys.stdin.readline().split())
        if T == 0:
            T0.append(X)
        elif T == 1:
            T1.append(X)
        else:
            T2.append(X)
    
    T0.sort(reverse=True)
    T1.sort(reverse=True)
    T2.sort(reverse=True)
    
    prefix_T0 = [0]
    for x in T0:
        prefix_T0.append(prefix_T0[-1] + x)
    prefix_T1 = [0]
    for x in T1:
        prefix_T1.append(prefix_T1[-1] + x)
    prefix_T2 = [0]
    for x in T2:
        prefix_T2.append(prefix_T2[-1] + x)
    
    global_max = 0
    max_k2 = min(M, len(T2))
    
    for k2 in range(0, max_k2 + 1):
        K = M - k2
        if K < 0:
            continue
        s2 = prefix_T2[k2]
        x_min = max(0, K - len(T0))
        x_max = min(s2, len(T1), K)
        if x_min > x_max:
            continue
        
        low, high, best_x = x_min, x_max, x_min
        while low <= high:
            mid = (low + high) // 2
            a_val = -float('inf')
            if mid > 0 and mid - 1 < len(T1):
                a_val = T1[mid - 1]
            t0_pos = K - mid - 1
            b_val = -float('inf')
            if 0 <= t0_pos < len(T0):
                b_val = T0[t0_pos]
            if a_val >= b_val:
                best_x = mid
                low = mid + 1
            else:
                high = mid - 1
        
        max_sum = 0
        for dx in [-1, 0, 1]:
            x = best_x + dx
            if x < x_min or x > x_max:
                continue
            t0 = K - x
            if 0 <= t0 <= len(T0):
                current_sum = prefix_T1[x] + prefix_T0[t0]
                if current_sum > max_sum:
                    max_sum = current_sum
        
        x = x_max
        t0 = K - x
        if 0 <= t0 <= len(T0):
            current_sum = prefix_T1[x] + prefix_T0[t0]
            if current_sum > max_sum:
                max_sum = current_sum
        x = x_min
        t0 = K - x
        if 0 <= t0 <= len(T0):
            current_sum = prefix_T1[x] + prefix_T0[t0]
            if current_sum > max_sum:
                max_sum = current_sum
        
        if max_sum > global_max:
            global_max = max_sum
    
    print(global_max)

if __name__ == '__main__':
    main()
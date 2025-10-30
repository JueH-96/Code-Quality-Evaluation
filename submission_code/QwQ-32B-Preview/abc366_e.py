import sys
import bisect

def compute_S(sorted_coords, prefix, suffix, k, y):
    N = len(sorted_coords)
    if y < sorted_coords[0]:
        k = 0
    elif y >= sorted_coords[-1]:
        k = N
    else:
        idx = bisect.bisect_right(sorted_coords, y)
        k = idx
    return (2 * k - N) * y + suffix[k] - prefix[k]

def find_y_min(sorted_y, prefix_y, suffix_y, T):
    left = sorted_y[0] - 10**6  # A safe lower bound
    right = sorted_y[-1] + 10**6  # A safe upper bound
    while left < right:
        mid = (left + right) // 2
        S_y = compute_S(sorted_y, prefix_y, suffix_y, 0, mid)
        if S_y <= T:
            right = mid
        else:
            left = mid + 1
    return left

def find_y_max(sorted_y, prefix_y, suffix_y, T):
    left = sorted_y[0] - 10**6  # A safe lower bound
    right = sorted_y[-1] + 10**6  # A safe upper bound
    while left < right:
        mid = (left + right + 1) // 2
        S_y = compute_S(sorted_y, prefix_y, suffix_y, 0, mid)
        if S_y <= T:
            left = mid
        else:
            right = mid - 1
    return left

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    x = list(map(int, data[2:2*N+2:2]))
    y = list(map(int, data[3:2*N+2:2]))
    
    sorted_x = sorted(x)
    sorted_y = sorted(y)
    
    prefix_x = [0] * (N + 1)
    suffix_x = [0] * (N + 1)
    for i in range(N):
        prefix_x[i+1] = prefix_x[i] + sorted_x[i]
        suffix_x[i+1] = suffix_x[i] + sorted_x[N-1-i]
    suffix_x = suffix_x[::-1]
    
    prefix_y = [0] * (N + 1)
    suffix_y = [0] * (N + 1)
    for i in range(N):
        prefix_y[i+1] = prefix_y[i] + sorted_y[i]
        suffix_y[i+1] = suffix_y[i] + sorted_y[N-1-i]
    suffix_y = suffix_y[::-1]
    
    total = 0
    for idx, x_val in enumerate(sorted_x):
        S_x = compute_S(sorted_x, prefix_x, suffix_x, idx, x_val)
        if D - S_x < 0:
            continue
        T = D - S_x
        y_min = find_y_min(sorted_y, prefix_y, suffix_y, T)
        y_max = find_y_max(sorted_y, prefix_y, suffix_y, T)
        if y_min > y_max:
            continue
        total += y_max - y_min + 1
    print(total)

if __name__ == '__main__':
    main()
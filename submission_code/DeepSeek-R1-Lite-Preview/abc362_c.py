import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    L = []
    R = []
    for _ in range(N):
        l = int(input[idx])
        r = int(input[idx + 1])
        L.append(l)
        R.append(r)
        idx += 2
    sum_min = sum(L)
    sum_max = sum(R)
    if sum_min > 0 or sum_max < 0:
        print("No")
        return
    # Initialize X to L
    X = L.copy()
    current_sum = sum_min
    if current_sum == 0:
        print("Yes")
        print(' '.join(map(str, X)))
        return
    elif current_sum < 0:
        need_increase = -current_sum
        # List of (increase possible, index)
        increases = []
        for i in range(N):
            if R[i] > L[i]:
                increases.append((R[i] - L[i], i))
        # Sort by possible increase descending
        increases.sort(reverse=True, key=lambda x: x[0])
        # Apply increases
        remaining = need_increase
        for inc, i in increases:
            if remaining <= 0:
                break
            add = min(inc, remaining)
            X[i] += add
            remaining -= add
        # Check if we achieved the required increase
        if remaining == 0:
            print("Yes")
            print(' '.join(map(str, X)))
        else:
            print("No")
        return
    else:
        # sum_min == 0
        print("Yes")
        print(' '.join(map(str, X)))
        return

if __name__ == '__main__':
    main()
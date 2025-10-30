import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    T = int(data[1])
    S = data[2].strip()
    X = list(map(int, data[3:]))

    # Pair up positions with directions, sort by position
    ants = list(zip(X, S))
    ants.sort(key=lambda x: x[0])

    # Extract sorted positions and a flag array:
    # flag = 1 if ant faces right ('1'), 0 if faces left ('0')
    P = [pos for pos, _ in ants]
    D = [1 if c == '1' else 0 for _, c in ants]

    # We need to count pairs (l, r) with l<r, D[l]=1, D[r]=0, and P[r] - P[l] <= 2*T.
    ans = 0
    count_right = 0  # number of right-facing ants in the current window
    left = 0
    limit = 2 * T

    for right in range(N):
        # Slide left up so that P[left] >= P[right] - limit
        threshold = P[right] - limit
        while left < right and P[left] < threshold:
            if D[left] == 1:
                count_right -= 1
            left += 1

        if D[right] == 1:
            # This ant is right-facing; it will count for future left-moving ants
            count_right += 1
        else:
            # This ant is left-facing; all right-facing ants in [left, right) will meet it
            ans += count_right

    print(ans)

if __name__ == "__main__":
    main()
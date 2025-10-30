import sys
import bisect

def main():
    data = sys.stdin.read().split()
    N = int(data[0]); T = int(data[1])
    S = data[2]
    Xs = list(map(int, data[3:3+N]))
    # Pair up positions with directions and sort by position
    ants = list(zip(Xs, S))
    ants.sort(key=lambda x: x[0])
    xs = [a[0] for a in ants]
    ss = [a[1] for a in ants]
    # Build prefix sums of zeros (ants facing negative)
    zero_psum = [0] * (N + 1)
    for i in range(N):
        zero_psum[i+1] = zero_psum[i] + (1 if ss[i] == '0' else 0)
    # For each ant facing right ('1'), count '0' ants to its right within distance 2*T
    ans = 0
    max_dist = 2 * T
    for i in range(N):
        if ss[i] == '1':
            # find largest index j such that xs[j] <= xs[i] + 2*T
            j = bisect.bisect_right(xs, xs[i] + max_dist) - 1
            if j > i:
                ans += zero_psum[j+1] - zero_psum[i+1]
    print(ans)

main()
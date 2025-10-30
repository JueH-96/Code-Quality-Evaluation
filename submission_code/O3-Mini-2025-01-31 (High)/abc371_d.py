import sys
import bisect

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    positions = [int(next(it)) for _ in range(N)]
    villagers = [int(next(it)) for _ in range(N)]
    
    # Build prefix sum array where prefix[i] stores the sum from positions[0] to positions[i-1]
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + villagers[i]
    
    Q = int(next(it))
    result = []
    for _ in range(Q):
        L = int(next(it))
        R = int(next(it))
        left_index = bisect.bisect_left(positions, L)
        right_index = bisect.bisect_right(positions, R)
        total = prefix[right_index] - prefix[left_index]
        result.append(str(total))
    
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()
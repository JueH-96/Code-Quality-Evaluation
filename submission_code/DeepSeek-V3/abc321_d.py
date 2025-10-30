# YOUR CODE HERE
import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    P = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    
    B_sorted = sorted(B)
    
    total = 0
    
    for a in A:
        # Find the number of b where a + b < P
        # So b < P - a
        threshold = P - a
        idx = bisect.bisect_left(B_sorted, threshold)
        # The sum for b < threshold is sum of b in B_sorted[:idx] + a * idx
        sum_below = sum(B_sorted[:idx])
        total += sum_below + a * idx
        # The sum for b >= threshold is P * (M - idx)
        total += P * (M - idx)
    
    print(total)

if __name__ == "__main__":
    main()
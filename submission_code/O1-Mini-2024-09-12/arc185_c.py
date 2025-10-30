# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from sys import stdin
    data = sys.stdin.read().split()
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:2+N]))
    indexed_A = [(A[i], i+1) for i in range(N)]
    indexed_A.sort()
    
    for l in range(N-2):
        a, idx_a = indexed_A[l]
        target = X - a
        left = l + 1
        right = N -1
        while left < right:
            b, idx_b = indexed_A[left]
            c, idx_c = indexed_A[right]
            total = b + c
            if total == target:
                # Ensure indices are in increasing order
                triplet = sorted([idx_a, idx_b, idx_c])
                print(triplet[0], triplet[1], triplet[2])
                return
            elif total < target:
                left +=1
            else:
                right -=1
    print(-1)

if __name__ == "__main__":
    main()
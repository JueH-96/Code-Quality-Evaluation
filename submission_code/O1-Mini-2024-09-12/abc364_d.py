# YOUR CODE HERE
import sys, bisect

def main():
    import sys, bisect
    input = sys.stdin.read().split()
    idx = 0
    N, Q = int(input[idx]), int(input[idx+1]); idx +=2
    A = list(map(int, input[idx:idx+N])); idx +=N
    A.sort()
    output = []
    for _ in range(Q):
        b_j, k_j = int(input[idx]), int(input[idx+1]); idx +=2
        left = 0
        right = 200000000
        while left < right:
            mid = (left + right) //2
            l = bisect.bisect_left(A, b_j - mid)
            r = bisect.bisect_right(A, b_j + mid)
            if r - l >= k_j:
                right = mid
            else:
                left = mid +1
        output.append(str(left))
    print('
'.join(output))

if __name__ == '__main__':
    main()
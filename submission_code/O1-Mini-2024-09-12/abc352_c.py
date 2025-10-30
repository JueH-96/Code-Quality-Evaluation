# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    sum_A = 0
    max_diff = -float('inf')
    idx = 1
    for _ in range(N):
        A = int(data[idx])
        B = int(data[idx+1])
        sum_A += A
        diff = B - A
        if diff > max_diff:
            max_diff = diff
        idx +=2
    print(sum_A + max_diff)

if __name__ == "__main__":
    main()
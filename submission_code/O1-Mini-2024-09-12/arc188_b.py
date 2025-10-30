# YOUR CODE HERE
import sys

def main():
    import sys
    import math
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    res = []
    for _ in range(T):
        N = int(input[idx])
        K = int(input[idx+1])
        idx +=2
        if N %2 ==0 and K*2 ==N:
            res.append("No")
        else:
            res.append("Yes")
    print('
'.join(res))

if __name__ == "__main__":
    main()
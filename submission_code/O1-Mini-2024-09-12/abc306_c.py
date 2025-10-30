# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    def input():
        return sys.stdin.read()
    
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    counts = [0] * (N + 1)
    f = [0] * (N + 1)
    
    for idx, num in enumerate(A, start=1):
        counts[num] +=1
        if counts[num] ==2:
            f[num] = idx
    
    sorted_nums = sorted(range(1, N+1), key=lambda x: f[x])
    print(' '.join(map(str, sorted_nums)))

if __name__ == "__main__":
    main()
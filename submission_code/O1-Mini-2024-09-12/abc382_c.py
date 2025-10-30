# YOUR CODE HERE
import sys

def main():
    import sys
    import threading
    def solve():
        N, M = map(int, sys.stdin.readline().split())
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
        
        MAX_X = 200000
        min_index = [N +1] * (MAX_X +2)
        
        for i in range(N-1, -1, -1):
            a = A[i]
            if a <= MAX_X:
                if min_index[a] > i +1:
                    min_index[a] = i +1
        
        for x in range(1, MAX_X +1):
            if min_index[x] > min_index[x -1]:
                min_index[x] = min_index[x -1]
        
        output = []
        for b in B:
            if b > MAX_X:
                b = MAX_X
            res = min_index[b]
            if res <= N:
                output.append(str(res))
            else:
                output.append("-1")
        print('
'.join(output))
    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()
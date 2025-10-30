# YOUR CODE HERE
import sys

def sum_abs_diff(arr):
    arr_sorted = sorted(arr)
    prefix_sum = 0
    total = 0
    for i, a in enumerate(arr_sorted):
        total += a * i - prefix_sum
        prefix_sum += a
    return total

def main():
    import sys
    import threading
    def run():
        data = sys.stdin.read().split()
        N = int(data[0])
        points = []
        idx = 1
        S0 = []
        S1 = []
        for _ in range(N):
            x = int(data[idx])
            y = int(data[idx+1])
            idx +=2
            if (x + y) %2 ==0:
                S0.append( (x, y) )
            else:
                S1.append( (x, y) )
        total = 0
        for S in [S0, S1]:
            if len(S) <2:
                continue
            u = [x + y for (x, y) in S]
            v = [x - y for (x, y) in S]
            sum_u = sum_abs_diff(u)
            sum_v = sum_abs_diff(v)
            total += (sum_u + sum_v) // 2
        print(total)
    threading.Thread(target=run).start()

if __name__ == "__main__":
    main()
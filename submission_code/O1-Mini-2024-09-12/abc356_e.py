# YOUR CODE HERE
import sys

def main():
    import sys
    import math
    input = sys.stdin.read
    data = list(map(int, input().split()))
    N = data[0]
    A = data[1:]
    A.sort()
    max_val = A[-1]
    freq = [0]*(max_val+2)
    for num in A:
        freq[num] +=1
    pre_sum = [0]*(max_val+2)
    for i in range(1, max_val+1):
        pre_sum[i] = pre_sum[i-1] + freq[i]
    total =0
    for x in range(1, max_val+1):
        if freq[x]==0:
            continue
        k =1
        while k*x <= max_val:
            l = k*x
            r = min((k+1)*x -1, max_val)
            cnt = pre_sum[r] - pre_sum[l-1]
            total += k * cnt * freq[x]
            k +=1
    # Subtract freq[x] for each x to remove pairs where i == j
    for x in range(1, max_val+1):
        if freq[x]:
            total -= freq[x]
    print(total)

if __name__ == "__main__":
    main()
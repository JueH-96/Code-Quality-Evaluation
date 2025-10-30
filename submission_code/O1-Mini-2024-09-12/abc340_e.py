# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from sys import stdin
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    A = [int(x) for x in data[idx:idx+N]]; idx +=N
    B = [int(x) for x in data[idx:idx+M]]; idx +=M
    add_all = 0
    extra_add = [0]*N
    for b in B:
        H_i = A[b] + add_all + extra_add[b]
        A[b] = -add_all - extra_add[b]
        c = H_i // N
        add_all += c
        k = H_i % N
        if k >0:
            start = (b +1) %N
            end = (b +k +1) %N
            if start < end:
                extra_add[start] +=1
                extra_add[end] -=1
            else:
                extra_add[start] +=1
                extra_add[0] +=1
                extra_add[end] -=1
    current =0
    for j in range(N):
        current += extra_add[j]
        A[j] += add_all + current
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()
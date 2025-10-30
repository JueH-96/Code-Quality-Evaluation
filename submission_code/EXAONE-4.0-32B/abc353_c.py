import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    total_sum = sum(A)
    total_pair_sum = (n - 1) * total_sum
    mod = 10**8
    
    A.sort()
    j = n - 1
    T = 0
    for i in range(n):
        while j > i and A[i] + A[j] >= mod:
            j -= 1
        T += (n - 1 - max(i, j))
    
    ans = total_pair_sum - mod * T
    print(ans)

if __name__ == "__main__":
    main()
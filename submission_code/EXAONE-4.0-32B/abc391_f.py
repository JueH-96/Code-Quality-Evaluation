import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+2*n]))
    C = list(map(int, data[2+2*n:2+3*n]))
    
    sorted_A = sorted(A)
    sorted_C = sorted(C)
    
    low = 0
    high = 3 * 10**18
    
    def count_ge(X):
        total_count = 0
        for bj in B:
            T = X + bj * bj
            ptr = n - 1
            cnt_j = 0
            for i in range(n):
                a_val = sorted_A[i] + bj
                while ptr >= 0:
                    c_val = sorted_C[ptr] + bj
                    if a_val * c_val >= T:
                        ptr -= 1
                    else:
                        break
                cnt_j += n - 1 - ptr
            total_count += cnt_j
        return total_count

    while low < high:
        mid = (low + high + 1) // 2
        if count_ge(mid) >= k:
            low = mid
        else:
            high = mid - 1
            
    print(low)

if __name__ == "__main__":
    main()
import sys

def main():
    N, L, R = map(int, sys.stdin.readline().split())
    
    def get_prefix_sum(X):
        if X < 0:
            return 0
        total = 0
        current = X
        while current >= 0:
            n_plus_1 = current + 1
            # Calculate the number of trailing zeros in n_plus_1
            i = 0
            temp = n_plus_1
            while temp % 2 == 0:
                i += 1
                temp //= 2
            size = 1 << i
            j_plus_1 = n_plus_1 // size
            j = j_plus_1 - 1
            # Query the judge
            print(f"? {i} {j}")
            sys.stdout.flush()
            T = int(sys.stdin.readline())
            if T == -1:
                exit()
            total = (total + T) % 100
            l = (1 << i) * j
            current = l - 1
        return total
    
    sum_R = get_prefix_sum(R)
    sum_L_minus_1 = get_prefix_sum(L - 1) if L > 0 else 0
    ans = (sum_R - sum_L_minus_1) % 100
    print(f"! {ans}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
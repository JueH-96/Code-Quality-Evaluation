import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    A = []
    B = []
    for _ in range(n):
        a = int(next(it))
        b = int(next(it))
        A.append(a)
        B.append(b)
    
    ans = 0
    for k in range(1, n + 1):
        found = False
        for last in range(n):
            other = []
            for i in range(n):
                if i == last:
                    continue
                other.append((A[i], B[i]))
            
            if k == 1:
                found = True
                break
                
            M = len(other)
            if M < k - 1:
                continue
                
            dp = [10**18] * (X + 1)
            dp[0] = 0
            
            for size_layer in range(1, k):
                new_dp = [10**18] * (X + 1)
                for a_i, b_i in other:
                    for a_val in range(X - a_i, -1, -1):
                        if dp[a_val] == 10**18:
                            continue
                        total_a = a_val + a_i
                        total_b = dp[a_val] + b_i
                        if total_b > Y:
                            continue
                        if total_b < new_dp[total_a]:
                            new_dp[total_a] = total_b
                dp = new_dp
                
            for a_val in range(X + 1):
                if dp[a_val] <= Y:
                    found = True
                    break
            if found:
                break
                
        if found:
            ans = k
            
    print(ans)

if __name__ == "__main__":
    main()
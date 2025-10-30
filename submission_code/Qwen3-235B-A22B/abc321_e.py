import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    results = []
    
    for _ in range(T):
        N = int(input[idx])
        X = int(input[idx+1])
        K = int(input[idx+2])
        idx +=3
        
        if K == 0:
            results.append("1")
            continue
        
        # Compute count_down(X, K, N)
        ans = 0
        Y = X
        d = K
        if Y == 0 or Y > N:
            temp = 0
        elif d == 0:
            temp = 1
        elif d >= 60:
            temp = 0
        else:
            min_val = Y << d
            if min_val > N:
                temp = 0
            else:
                max_val = ((Y + 1) << d) -1
                if max_val > N:
                    temp = N - min_val +1
                else:
                    temp = max_val - min_val +1
        ans = temp
        
        max_a = min(K, 60)
        for a in range(1, max_a +1):
            A = X >> a
            if A == 0:
                break
            s = K - a
            if s < 0:
                continue
            if s == 0:
                ans += 1
            else:
                # compute other_child
                bit = (X >> (a-1)) & 1
                other_child = A * 2 + (bit ^ 1)
                if other_child > N:
                    continue
                # compute count_down(other_child, s-1, N)
                Y_oc = other_child
                d_oc = s -1
                if Y_oc ==0 or Y_oc > N:
                    temp_oc =0
                elif d_oc ==0:
                    temp_oc =1
                elif d_oc >=60:
                    temp_oc =0
                else:
                    min_val_oc = Y_oc << d_oc
                    if min_val_oc > N:
                        temp_oc =0
                    else:
                        max_val_oc = ((Y_oc +1) << d_oc) -1
                        if max_val_oc > N:
                            temp_oc = N - min_val_oc +1
                        else:
                            temp_oc = max_val_oc - min_val_oc +1
                ans += temp_oc
        results.append(str(ans))
    
    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()
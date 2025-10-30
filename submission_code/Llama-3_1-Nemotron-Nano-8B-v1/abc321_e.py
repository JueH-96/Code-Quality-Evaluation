import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx +=1
    results = []
    for _ in range(T):
        N = int(input[idx])
        X = int(input[idx+1])
        K = int(input[idx+2])
        idx +=3
        
        D = X.bit_length() -1
        ans =0
        
        max_a = min(K, D)
        for a in range(max_a +1):
            l = X >> a
            s = K - a
            if s <0:
                continue
            # Compute s_max for l
            if l ==0:
                continue
            temp = N // l
            if temp ==0:
                s_max = -1
            else:
                s_max = temp.bit_length() -1
            if s > s_max:
                continue
            min_node = l << s
            if min_node > N:
                continue
            max_node = min_node + (1 << s) -1
            if max_node > N:
                cnt = N - min_node +1
            else:
                cnt = 1 << s
            ans += cnt
        
        results.append(ans)
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == '__main__':
    main()
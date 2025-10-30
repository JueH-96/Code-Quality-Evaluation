import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        N = int(input[idx])
        X = int(input[idx+1])
        K = int(input[idx+2])
        idx +=3
        
        if K == 0:
            print(1)
            continue
        
        depth_X = X.bit_length() - 1
        max_a = min(K, depth_X)
        total = 0
        
        for a in range(max_a + 1):
            rem = K - a
            if rem < 0:
                continue
            A = X >> a
            if rem == 0:
                total += 1
                continue
            
            if a == 0:
                root = X
                m = rem
            else:
                P = X >> (a-1)
                if (A << 1) == P:
                    B = (A << 1) + 1
                else:
                    B = A << 1
                if B > N:
                    continue
                root = B
                m = rem - 1
            
            if m < 0:
                continue
            
            try:
                min_node = root * (1 << m)
            except:
                min_node = root * (2 ** m)
            
            if min_node > N:
                continue
            
            try:
                max_node = min_node + (1 << m) - 1
            except:
                max_node = min_node + (2 ** m) - 1
            
            if max_node > N:
                count = N - min_node + 1
            else:
                count = 1 << m
            
            total += count
        
        print(total)

if __name__ == '__main__':
    main()
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    idx = 1
    for _ in range(T):
        K = int(data[idx])
        Xs = int(data[idx+1])
        Ys = int(data[idx+2])
        Ts = int(data[idx+3])
        Xe = int(data[idx+4])
        Ye = int(data[idx+5])
        idx += 6
        
        # Compute is and js
        is_s = (Xs + K) // K
        js_s = (Ys + K) // K
        
        # Compute it and jt
        it_t = (Xe + K) // K
        jt_t = (Ye + K) // K
        
        # Compute ks and kt
        ks = (Ys - js_s * K) // 1
        kt = (Ye - jt_t * K) // 1
        
        D = abs(it_t - is_s) + abs(jt_t - js_s)
        dk = abs(kt - ks)
        
        steps = D + dk
        if (D + dk) % 2 == 1:
            steps += 1
        
        print(steps)

if __name__ == '__main__':
    main()
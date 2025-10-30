import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        N = int(data[index]); index += 1
        P = list(map(int, data[index:index+N]))
        index += N
        
        is_sorted = True
        for i in range(N):
            if P[i] != i+1:
                is_sorted = False
                break
                
        if is_sorted:
            results.append("0")
            continue
            
        prefix_min = [0] * N
        prefix_max = [0] * N
        prefix_min[0] = P[0]
        prefix_max[0] = P[0]
        for i in range(1, N):
            prefix_min[i] = min(prefix_min[i-1], P[i])
            prefix_max[i] = max(prefix_max[i-1], P[i])
            
        suffix_min = [0] * N
        suffix_max = [0] * N
        suffix_min[N-1] = P[N-1]
        suffix_max[N-1] = P[N-1]
        for i in range(N-2, -1, -1):
            suffix_min[i] = min(suffix_min[i+1], P[i])
            suffix_max[i] = max(suffix_max[i+1], P[i])
            
        found_one = False
        for k in range(1, N+1):
            if P[k-1] != k:
                continue
                
            left_ok = True
            if k >= 2:
                if prefix_min[k-2] != 1 or prefix_max[k-2] != k-1:
                    left_ok = False
                    
            if not left_ok:
                continue
                
            right_ok = True
            if k <= N-1:
                if suffix_min[k] != k+1 or suffix_max[k] != N:
                    right_ok = False
                    
            if not right_ok:
                continue
                
            found_one = True
            break
            
        if found_one:
            results.append("1")
        else:
            results.append("2")
            
    print("
".join(results))

if __name__ == '__main__':
    main()
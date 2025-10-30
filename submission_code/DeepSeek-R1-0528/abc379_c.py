import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    X = [int(next(it)) for _ in range(M)]
    A = [int(next(it)) for _ in range(M)]
    
    total_stones = sum(A)
    if total_stones != N:
        print(-1)
        return
        
    if M == 0:
        print(-1)
        return
        
    stones = []
    for i in range(M):
        stones.append((X[i], A[i]))
    stones.sort()
    
    if stones[0][0] != 1:
        print(-1)
        return
        
    cum = 0
    S_list = []
    for i in range(M):
        x, a = stones[i]
        cum += a
        S_list.append(cum)
        if cum < x:
            print(-1)
            return
        if i < M - 1:
            next_x = stones[i+1][0]
            if cum < next_x - 1:
                print(-1)
                return
                
    total_ops = 0
    for k in range(M):
        x, a_val = stones[k]
        L = x
        if k < M - 1:
            next_x = stones[k+1][0]
            R = min(next_x - 1, N - 1)
        else:
            R = N - 1
            
        if L > R:
            continue
            
        len_seg = R - L + 1
        total_ops += S_list[k] * len_seg - (L + R) * len_seg // 2
        
    print(total_ops)

if __name__ == "__main__":
    main()
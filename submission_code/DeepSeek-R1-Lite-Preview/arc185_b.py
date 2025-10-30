def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = list(map(int, data[index:index + N]))
        index += N
        
        S = [0] * N
        S[0] = A[0]
        for i in range(1, N):
            S[i] = S[i - 1] + A[i]
        
        T_seq = [0] * N
        T_seq[0] = S[0]
        for i in range(1, N):
            T_seq[i] = max(S[i], T_seq[i - 1])
        
        sum_A = S[-1]
        sum_B = T_seq[-1]
        
        if sum_B == sum_A:
            results.append("Yes")
        else:
            results.append("No")
    
    print("
".join(results))

if __name__ == "__main__":
    main()
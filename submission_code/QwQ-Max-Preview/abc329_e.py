def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    S = input[idx]
    idx += 1
    T = input[idx]
    
    pre_max_i = [-1] * N
    for j in range(N):
        max_i = -1
        for k in range(M):
            i_candidate = j - k
            if i_candidate < 0 or i_candidate > N - M:
                continue
            if T[k] == S[j]:
                if i_candidate > max_i:
                    max_i = i_candidate
        if max_i == -1:
            print("No")
            return
        pre_max_i[j] = max_i
    
    ops = set(pre_max_i)
    for i in ops:
        for k in range(M):
            j = i + k
            if j >= N:
                break
            if pre_max_i[j] < i:
                print("No")
                return
    print("Yes")

if __name__ == "__main__":
    main()
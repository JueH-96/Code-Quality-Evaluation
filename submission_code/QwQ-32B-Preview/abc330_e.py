def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    freq = [0] * (N + 1)
    for num in A:
        if num <= N:
            freq[num] += 1
    
    current_mex = 0
    while current_mex <= N and freq[current_mex] > 0:
        current_mex += 1
    
    index = 2 + N
    for _ in range(Q):
        i_k = int(data[index]) - 1  # Convert to 0-based index
        x_k = int(data[index + 1])
        index += 2
        
        old_value = A[i_k]
        if old_value <= N:
            freq[old_value] -= 1
            if old_value < current_mex and freq[old_value] == 0:
                current_mex = old_value
        A[i_k] = x_k
        if x_k <= N:
            freq[x_k] += 1
            if x_k < current_mex:
                if x_k == current_mex - 1:
                    current_mex += 1
                    while current_mex <= N and freq[current_mex] > 0:
                        current_mex += 1
                elif x_k == current_mex:
                    current_mex += 1
                    while current_mex <= N and freq[current_mex] > 0:
                        current_mex += 1
            elif x_k > current_mex:
                if freq[current_mex] > 0:
                    while current_mex <= N and freq[current_mex] > 0:
                        current_mex += 1
        print(current_mex)

if __name__ == '__main__':
    main()
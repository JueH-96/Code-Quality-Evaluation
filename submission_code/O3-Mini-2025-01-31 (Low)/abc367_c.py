def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])
    R = [int(x) for x in input_data[2:2+N]]
    
    result = []
    seq = [0] * N
    
    def dfs(idx, current_sum):
        if idx == N:
            if current_sum % K == 0:
                # print current sequence to output
                result.append(" ".join(map(str, seq)))
            return
        
        # iterate from 1 to R[idx], inclusive. This order is lexicographical.
        for value in range(1, R[idx] + 1):
            seq[idx] = value
            dfs(idx + 1, current_sum + value)
    
    dfs(0, 0)
    
    sys.stdout.write("
".join(result))
    
if __name__ == '__main__':
    main()
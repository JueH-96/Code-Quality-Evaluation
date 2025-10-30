import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    results = []
    for _ in range(T):
        N = int(input[idx])
        idx += 1
        A = input[idx]
        idx += 1
        B = input[idx]
        idx += 1
        
        cntA = A.count('1')
        cntB = B.count('1')
        if cntB > cntA:
            results.append("-1")
            continue
        
        a_list = []
        for i in range(N):
            if A[i] == '1':
                a_list.append(i+1)
        
        b_list = []
        for i in range(N):
            if B[i] == '1':
                b_list.append(i+1)
        
        m = len(b_list)
        # Take first m elements of a_list
        a_sub = a_list[:m]
        max_ops = 0
        for i in range(m):
            diff = abs(a_sub[i] - b_list[i])
            if diff > max_ops:
                max_ops = diff
        results.append(str(max_ops))
    
    print('
'.join(results))

if __name__ == '__main__':
    main()
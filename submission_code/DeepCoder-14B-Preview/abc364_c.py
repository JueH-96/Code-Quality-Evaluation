def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    X = int(input[idx]); idx +=1
    Y = int(input[idx]); idx +=1
    
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N
    
    dishes = list(zip(A, B))
    
    # Check if any single dish can cause immediate stop
    for a, b in dishes:
        if a > X or b > Y:
            print(1)
            return
    
    # Sort by A descending and compute prefix sum
    sorted_a = sorted(dishes, key=lambda x: -x[0])
    sum_a = 0
    min_k_a = None
    for i in range(len(sorted_a)):
        sum_a += sorted_a[i][0]
        if sum_a > X:
            min_k_a = i + 1
            break
    
    # Sort by B descending and compute prefix sum
    sorted_b = sorted(dishes, key=lambda x: -x[1])
    sum_b = 0
    min_k_b = None
    for i in range(len(sorted_b)):
        sum_b += sorted_b[i][1]
        if sum_b > Y:
            min_k_b = i + 1
            break
    
    candidates = []
    if min_k_a is not None:
        candidates.append(min_k_a)
    if min_k_b is not None:
        candidates.append(min_k_b)
    
    if not candidates:
        print(N)
    else:
        print(min(candidates))

if __name__ == '__main__':
    main()
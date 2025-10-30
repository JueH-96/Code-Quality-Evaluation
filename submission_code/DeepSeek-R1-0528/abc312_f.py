import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    M = int(data[1])
    items = []
    idx = 2
    for i in range(n):
        t = int(data[idx])
        x = int(data[idx+1])
        idx += 2
        items.append((t, x))
        
    A = []
    B = []
    C = []
    for t, x in items:
        if t == 0:
            A.append(x)
        elif t == 1:
            B.append(x)
        else:
            C.append(x)
            
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)
    
    prefixA = [0]
    for x in A:
        prefixA.append(prefixA[-1] + x)
        
    prefixB = [0]
    for x in B:
        prefixB.append(prefixB[-1] + x)
        
    prefixC = [0]
    for x in C:
        prefixC.append(prefixC[-1] + x)
        
    max_b = min(M, len(B))
    k = 0
    ans = 0
    for b in range(0, max_b + 1):
        while k <= len(C) and prefixC[k] < b:
            k += 1
        if k > len(C):
            break
        c_min = k
        if c_min > M - b:
            continue
        a_val = M - b - c_min
        pull_sum = prefixA[min(a_val, len(A))]
        regular_sum = prefixB[b]
        total = pull_sum + regular_sum
        if total > ans:
            ans = total
            
    print(ans)

if __name__ == "__main__":
    main()
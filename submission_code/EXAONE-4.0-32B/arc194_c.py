import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    B = list(map(int, data[1+n:1+2*n]))
    C = list(map(int, data[1+2*n:1+3*n]))
    
    diff = [a != b for a, b in zip(A, B)]
    M = sum(diff)
    
    base = 0
    list1 = []
    list2 = []
    
    for i in range(n):
        if not diff[i]:
            base += M * A[i] * C[i]
            
    for i in range(n):
        if diff[i]:
            if A[i] == 1:
                list1.append(C[i])
            else:
                list2.append(C[i])
            base += (-A[i] * (M + 2) + (M + 1)) * C[i]
            
    list1.sort(reverse=True)
    list2.sort(reverse=True)
    
    T = 0
    for idx, c_val in enumerate(list1):
        t_i = idx + 1
        T += t_i * c_val
        
    for idx, c_val in enumerate(list2):
        t_i = M - idx
        T -= t_i * c_val
        
    total_cost = base + T
    print(total_cost)

if __name__ == "__main__":
    main()
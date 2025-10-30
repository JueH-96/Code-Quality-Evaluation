import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    points = []
    index = 1
    for i in range(n):
        x = int(data[index])
        y = int(data[index + 1])
        index += 2
        points.append((x, y))
    
    group0 = []
    group1 = []
    
    for x, y in points:
        if (x + y) % 2 == 0:
            group0.append((x, y))
        else:
            group1.append((x, y))
            
    def compute_T(arr):
        arr.sort()
        n = len(arr)
        total = 0
        for j in range(n):
            total += arr[j] * (2 * j - n + 1)
        return total
        
    total_sum = 0
    
    if group0:
        U0 = [x + y for x, y in group0]
        V0 = [x - y for x, y in group0]
        T0_u = compute_T(U0)
        T0_v = compute_T(V0)
        T0 = T0_u + T0_v
        total_sum += T0 // 2
        
    if group1:
        U1 = [x + y for x, y in group1]
        V1 = [x - y for x, y in group1]
        T1_u = compute_T(U1)
        T1_v = compute_T(V1)
        T1 = T1_u + T1_v
        total_sum += T1 // 2
        
    print(total_sum)

if __name__ == "__main__":
    main()
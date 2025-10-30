def nand(a, b):
    return 1 if (a == 0 and b == 0) or (a == 0 and b == 1) or (a == 1 and b == 0) else 0

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    
    A = [int(c) for c in S]
    
    total_sum = 0
    
    for i in range(N):
        current_value = A[i]
        for j in range(i, N):
            if i < j:
                current_value = nand(current_value, A[j])
            total_sum += current_value
    
    print(total_sum)

if __name__ == "__main__":
    main()
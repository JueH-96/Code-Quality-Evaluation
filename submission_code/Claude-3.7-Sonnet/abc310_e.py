def main():
    N = int(input())
    S = input()
    A = [int(c) for c in S]
    
    total_sum = 0
    for i in range(N):
        current_f = A[i]
        total_sum += current_f
        for j in range(i+1, N):
            # NAND operation: returns 0 only when both inputs are 1, otherwise returns 1
            current_f = 0 if current_f == 1 and A[j] == 1 else 1
            total_sum += current_f
    
    print(total_sum)

if __name__ == "__main__":
    main()
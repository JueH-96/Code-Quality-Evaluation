def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    total = sum(A)
    r = S % total
    
    current_sum = 0
    seen_sums = {0}
    for i in range(2 * N):
        current_sum += A[i % N]
        if current_sum - r in seen_sums:
            print("Yes")
            return
        seen_sums.add(current_sum)
    print("No")

if __name__ == "__main__":
    main()
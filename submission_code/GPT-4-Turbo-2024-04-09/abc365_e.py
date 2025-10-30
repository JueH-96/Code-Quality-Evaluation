def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    result = 0
    for i in range(N):
        xor_sum = 0
        for j in range(i, N):
            xor_sum ^= A[j]
            result += xor_sum
    
    print(result)

if __name__ == "__main__":
    main()
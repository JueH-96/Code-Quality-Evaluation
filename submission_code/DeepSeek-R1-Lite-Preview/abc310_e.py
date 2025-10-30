def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    A = [int(ch) for ch in S]
    
    prev_dp1 = 0
    prev_dp0 = 0
    total_sum = 0
    for j in range(1, N+1):
        if A[j-1] == 0:
            current_dp1 = j - 1
            current_dp0 = 1
        else:
            current_dp1 = prev_dp0 + 1
            current_dp0 = prev_dp1
        total_sum += current_dp1
        prev_dp1 = current_dp1
        prev_dp0 = current_dp0
    print(total_sum)

if __name__ == "__main__":
    main()
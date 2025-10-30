def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    sum_total = 0
    sum_j = 0
    for j in range(1, N + 1):
        digit = int(S[j - 1])
        sum_j = sum_j * 10 + digit * j
        sum_total += sum_j
    print(sum_total)

if __name__ == "__main__":
    main()
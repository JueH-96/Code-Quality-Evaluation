def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    sum_A = sum(A)
    total = 0
    for bit in range(31):
        current_pfx = 0
        count0 = 1
        count1 = 0
        for num in A:
            bit_val = (num >> bit) & 1
            current_pfx ^= bit_val
            if current_pfx == 0:
                opposite = count1
            else:
                opposite = count0
            total += opposite * (1 << bit)
            if current_pfx == 0:
                count0 += 1
            else:
                count1 += 1
    answer = total - sum_A
    print(answer)

if __name__ == "__main__":
    main()
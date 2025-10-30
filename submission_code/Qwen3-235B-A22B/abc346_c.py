def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:2+N]))
    
    present = set()
    for x in A:
        if x <= K:
            present.add(x)
    sum_present = sum(present)
    total = K * (K + 1) // 2
    print(total - sum_present)

if __name__ == "__main__":
    main()
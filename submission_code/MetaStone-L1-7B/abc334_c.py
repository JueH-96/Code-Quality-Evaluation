def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:2+N]))
    
    two = []
    single = []
    for a in A:
        if a == 1:
            single.append(1)
        elif a == 2 and A.count(a) >= K:
            two.append(a)
        else:
            single.append(a)
    
    # Sort both lists
    two_sorted = sorted(two)
    single_sorted = sorted(single)
    
    total = 0
    # Pair in two_sorted
    for i in range(0, len(two_sorted), 2):
        if i + 1 < len(two_sorted):
            total += two_sorted[i+1] - two_sorted[i]
    # Pair in single_sorted
    for i in range(0, len(single_sorted), 2):
        if i + 1 < len(single_sorted):
            total += single_sorted[i+1] - single_sorted[i]
    
    print(total)

if __name__ == "__main__":
    main()
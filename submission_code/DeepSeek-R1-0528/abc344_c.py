def main():
    n = int(input())
    A = list(map(int, input().split()))
    m = int(input())
    B = list(map(int, input().split()))
    l = int(input())
    C = list(map(int, input().split()))
    q = int(input())
    X = list(map(int, input().split()))
    
    total_sums = set()
    for a in A:
        for b in B:
            for c in C:
                total_sums.add(a + b + c)
                
    for x in X:
        print("Yes" if x in total_sums else "No")

if __name__ == "__main__":
    main()
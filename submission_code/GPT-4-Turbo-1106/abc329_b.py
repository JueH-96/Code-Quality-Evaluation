def main():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))
    
    max_val = max(A)
    second_largest = max(x for x in A if x != max_val)
    
    print(second_largest)

if __name__ == "__main__":
    main()
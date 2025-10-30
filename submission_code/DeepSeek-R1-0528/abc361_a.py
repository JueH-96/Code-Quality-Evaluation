def main():
    data = input().split()
    n = int(data[0])
    k = int(data[1])
    x = int(data[2])
    A = list(map(int, input().split()))
    
    part1 = A[:k]
    part2 = A[k:]
    B = part1 + [x] + part2
    
    print(" ".join(map(str, B)))

if __name__ == "__main__":
    main()
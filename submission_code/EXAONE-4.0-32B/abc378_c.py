def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    last_occurrence = {}
    result = []
    for i in range(n):
        num = A[i]
        if num in last_occurrence:
            result.append(str(last_occurrence[num] + 1))
        else:
            result.append("-1")
        last_occurrence[num] = i
    
    print(" ".join(result))

if __name__ == "__main__":
    main()
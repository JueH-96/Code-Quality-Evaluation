def main():
    n = int(input().strip())
    A = list(map(int, input().split()))
    
    last_occurrence = {}
    B = []
    
    for i in range(n):
        num = A[i]
        if num in last_occurrence:
            B.append(last_occurrence[num] + 1)
        else:
            B.append(-1)
        last_occurrence[num] = i
        
    print(" ".join(map(str, B)))

if __name__ == '__main__':
    main()
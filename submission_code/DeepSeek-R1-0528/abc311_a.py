def main():
    n = int(input().strip())
    s = input().strip()
    
    countA = 0
    countB = 0
    countC = 0
    
    for i in range(n):
        char = s[i]
        if char == 'A':
            countA += 1
        elif char == 'B':
            countB += 1
        elif char == 'C':
            countC += 1
            
        if countA >= 1 and countB >= 1 and countC >= 1:
            print(i + 1)
            return

if __name__ == '__main__':
    main()
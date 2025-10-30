def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    t = int(data[0])
    index = 1
    
    results = []
    for _ in range(t):
        S = data[index]
        X = data[index + 1]
        Y = data[index + 2]
        index += 3
        
        if len(S) == 0:
            results.append("Yes")
            continue
        
        countX0 = X.count('0')
        countX1 = X.count('1')
        countY0 = Y.count('0')
        countY1 = Y.count('1')
        
        a = countY0 - countX0
        b = countX1 - countY1
        lenS = len(S)
        
        if b != 0:
            if a * lenS % b != 0:
                results.append("No")
            else:
                lenT = (a * lenS) // b
                if lenT >= 0:
                    results.append("Yes")
                else:
                    results.append("No")
        else:
            if a * lenS == 0 and countX1 == countY1:
                results.append("Yes")
            else:
                results.append("No")
    
    print("
".join(results))

if __name__ == "__main__":
    main()
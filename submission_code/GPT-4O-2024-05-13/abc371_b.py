# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    families = {}
    index = 2
    
    results = []
    
    for _ in range(M):
        family = int(data[index])
        gender = data[index + 1]
        index += 2
        
        if gender == 'M':
            if family not in families:
                families[family] = True
                results.append("Yes")
            else:
                results.append("No")
        else:
            results.append("No")
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
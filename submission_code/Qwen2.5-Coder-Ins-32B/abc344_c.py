# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    
    index = 0
    N = int(input[index])
    A = list(map(int, input[index+1:index+1+N]))
    index += 1 + N
    
    M = int(input[index])
    B = list(map(int, input[index+1:index+1+M]))
    index += 1 + M
    
    L = int(input[index])
    C = list(map(int, input[index+1:index+1+L]))
    index += 1 + L
    
    Q = int(input[index])
    X = list(map(int, input[index+1:index+1+Q]))
    
    possible_sums = set()
    for a in A:
        for b in B:
            possible_sums.add(a + b)
    
    results = []
    for x in X:
        found = False
        for c in C:
            if (x - c) in possible_sums:
                found = True
                break
        results.append("Yes" if found else "No")
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()